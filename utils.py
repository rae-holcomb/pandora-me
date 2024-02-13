import numpy as np
import math

# Standard library
import os
import warnings
# from functools import lru_cache

# Third-party
import astropy.units as u
import numpy as np
from astropy.coordinates import Distance, SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia
from astropy.stats import sigma_clip
from astropy import table
from astropy.wcs import WCS, Sip
import astropy.units as u
from astropy.io import fits

from scipy import sparse
from typing import Optional, Tuple
from copy import deepcopy

# to be removed later
import lightkurve as lk



## HELPER FUNCTIONS
# Currently set up for TESS not Pandora

def fits_from_table(table, header=None):
    """Helper function to convert astropy.Table to astropy.fits.TableHDU"""
    cols = [
        fits.Column(col.name, col.dtype, unit=col.unit.name if col.unit is not None else None, array=col.data)
        for col in table.itercols()
    ]
    return fits.TableHDU.from_columns(cols, nrows=len(table), header=header)


def gaussian_2d(x, y, mu_x, mu_y, sigma_x=2, sigma_y=2):
    """
    Compute the value of a 2D Gaussian function.

    Parameters:
    x (float): x-coordinate.
    y (float): y-coordinate.
    mu_x (float): Mean of the Gaussian in the x-direction.
    mu_y (float): Mean of the Gaussian in the y-direction.
    sigma_x (float): Standard deviation of the Gaussian in the x-direction.
    sigma_y (float): Standard deviation of the Gaussian in the y-direction.

    Returns:
    float: Value of the 2D Gaussian function at (x, y).
    """
    part1 = 1 / (2 * np.pi * sigma_x * sigma_y)
    part2 = np.exp(
        -((x - mu_x) ** 2 / (2 * sigma_x**2) + (y - mu_y) ** 2 / (2 * sigma_y**2))
    )
    return part1 * part2

def build_A_matrix(M1, M2, polyorder1=1, polyorder2=1):
    """Helps build a design matrix that is the linear combination of two 2D matrices M1 and M2. polyorder 1 and 2 set the order of M1 and M2 respectively and can be an integer or an array."""
    # allow for integers or arrays for the polyorders
    if isinstance(polyorder1, int):
        polyorder1 = np.arange(polyorder1+1)
    if isinstance(polyorder2, int):
        polyorder2 = np.arange(polyorder2+1)
    
    # Design matrix
    A = np.vstack(
        [
            M1.ravel() ** idx * M2.ravel() ** jdx
            for idx in polyorder1
            for jdx in polyorder2
        ]
    ).T

    return A


def get_wcs(
    detector,
    # detector: Detector,
    target_ra: u.Quantity,
    target_dec: u.Quantity,
    crpix1: int = None,
    crpix2: int = None,
    theta: u.Quantity = 0 * u.deg,
    distortion_file: str = None,
    order: int = 3,
    xreflect: bool = True,
    yreflect: bool = False,    
) -> WCS.wcs:
    """Get the World Coordinate System for a detector

    Parameters:
    -----------
    detector : pandorasim.Detector
        The detector to build the WCS for
    target_ra: astropy.units.Quantity
        The target RA in degrees
    target_dec: astropy.units.Quantity
        The target Dec in degrees
    theta: astropy.units.Quantity
        The observatory angle in degrees
    distortion_file: str
        Optional file path to a distortion CSV file. See `read_distortion_file`
    """
    # xreflect = True
    # yreflect = False
    hdu = fits.PrimaryHDU()
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    matrix = np.asarray(
        [
            [np.cos(theta).value, -np.sin(theta).value],
            [np.sin(theta).value, np.cos(theta).value],
        ]
    )
    hdu.header["CRVAL1"] = target_ra.value
    hdu.header["CRVAL2"] = target_dec.value
    for idx in range(2):
        for jdx in range(2):
            hdu.header[f"PC{idx+1}_{jdx+1}"] = matrix[idx, jdx]
    hdu.header["CRPIX1"] = (
        detector.naxis1.value // 2 if crpix1 is None else crpix1
    )
    hdu.header["CRPIX2"] = (
        detector.naxis2.value // 2 if crpix2 is None else crpix2
    )
    hdu.header["NAXIS1"] = detector.naxis1.value
    hdu.header["NAXIS2"] = detector.naxis2.value
    hdu.header["CDELT1"] = detector.pixel_scale.to(u.deg / u.pixel).value * (
        -1
    ) ** (int(xreflect))
    hdu.header["CDELT2"] = detector.pixel_scale.to(u.deg / u.pixel).value * (
        -1
    ) ** (int(yreflect))
    if distortion_file is not None:
        # wcs = _get_distorted_wcs(
        #     detector, hdu.header, distortion_file, order=order
        # )
        pass
    else:
        wcs = WCS(hdu.header)
    return wcs

# copied in from pandorasim but added in gmag and gflux
def get_sky_catalog(
    ra=210.8023,
    dec=54.349,
    radius=0.155,
    gbpmagnitude_range=(-3, 20),
    limit=None,
    gaia_keys=[],
    time: Time =Time.now()
) -> dict :
    """Gets a catalog of coordinates on the sky based on an input ra, dec and radius
    
    Gaia keys will add in additional keywords to be grabbed from Gaia catalog."""

    base_keys = ["source_id",
        "ra",
        "dec",
        "parallax",
        "pmra",
        "pmdec",
        "radial_velocity",
        "ruwe",
        "phot_bp_mean_mag",
        "teff_gspphot",
        "logg_gspphot",
        "phot_g_mean_flux", 
        "phot_g_mean_mag",]

    all_keys = base_keys + gaia_keys

    query_str = f"""
    SELECT {f'TOP {limit} ' if limit is not None else ''}* FROM (
        SELECT gaia.{', gaia.'.join(all_keys)}, dr2.teff_val AS dr2_teff_val,
        dr2.rv_template_logg AS dr2_logg, tmass.j_m, tmass.j_msigcom, tmass.ph_qual, DISTANCE(
        POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
        POINT(gaia.ra, gaia.dec)) AS ang_sep,
        EPOCH_PROP_POS(gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec,
        gaia.radial_velocity, gaia.ref_epoch, 2000) AS propagated_position_vector
        FROM gaiadr3.gaia_source AS gaia
        JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id)
        JOIN gaiadr3.dr2_neighbourhood AS xmatch2 ON gaia.source_id = xmatch2.dr3_source_id
        JOIN gaiadr2.gaia_source AS dr2 ON xmatch2.dr2_source_id = dr2.source_id
        JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid)
        JOIN gaiadr1.tmass_original_valid AS tmass ON
        xjoin.original_psc_source_id = tmass.designation
        WHERE 1 = CONTAINS(
        POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
        CIRCLE(gaia.ra, gaia.dec, {(u.Quantity(radius, u.deg) + 50*u.arcsecond).value}))
        AND gaia.parallax IS NOT NULL
        AND gaia.phot_bp_mean_mag > {gbpmagnitude_range[0]}
        AND gaia.phot_bp_mean_mag < {gbpmagnitude_range[1]}) AS subquery
    WHERE 1 = CONTAINS(
    POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
    CIRCLE(COORD1(subquery.propagated_position_vector), COORD2(subquery.propagated_position_vector), {u.Quantity(radius, u.deg).value}))
    ORDER BY ang_sep ASC
    """
    job = Gaia.launch_job_async(query_str, verbose=False)
    tbl = job.get_results()
    if len(tbl) == 0:
        raise ValueError("Could not find matches.")
    plx = tbl["parallax"].value.filled(fill_value=0)
    plx[plx < 0] = 0
    cat = {
        "jmag": tbl["j_m"].data.filled(np.nan),
        "bmag": tbl["phot_bp_mean_mag"].data.filled(np.nan),
        "gmag": tbl["phot_g_mean_mag"].data.filled(np.nan),
        "gflux": tbl["phot_g_mean_flux"].data.filled(np.nan),
        "ang_sep": tbl["ang_sep"].data.filled(np.nan) * u.deg,
    }
    cat["teff"] = (
        tbl["teff_gspphot"].data.filled(
            tbl["dr2_teff_val"].data.filled(np.nan)
        )
        * u.K
    )
    cat["logg"] = tbl["logg_gspphot"].data.filled(
        tbl["dr2_logg"].data.filled(np.nan)
    )
    cat["RUWE"] = tbl["ruwe"].data.filled(99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat["coords"] = SkyCoord(
            ra=tbl["ra"].value.data * u.deg,
            dec=tbl["dec"].value.data * u.deg,
            pm_ra_cosdec=tbl["pmra"].value.filled(fill_value=0)
            * u.mas
            / u.year,
            pm_dec=tbl["pmdec"].value.filled(fill_value=0) * u.mas / u.year,
            obstime=Time.strptime("2016", "%Y"),
            distance=Distance(parallax=plx * u.mas, allow_negative=True),
            radial_velocity=tbl["radial_velocity"].value.filled(fill_value=0)
            * u.km
            / u.s,
        ).apply_space_motion(time)
    cat["source_id"] = np.asarray(
        [f"Gaia DR3 {i}" for i in tbl["source_id"].value.data]
    )
    for key in gaia_keys:
        cat[key] = tbl[key].data.filled(np.nan)
    return cat


def fit_bkg(tpf: lk.TessTargetPixelFile, polyorder: int = 1) -> np.ndarray:
    """Fit a simple 2d polynomial background to a TPF

    Parameters
    ----------
    tpf: lightkurve.TessTargetPixelFile
        Target pixel file object
    polyorder: int
        Polynomial order for the model fit.

    Returns
    -------
    model : np.ndarray
        Model for background with same shape as tpf.shape
    """
    # Notes for understanding this function
    # All arrays in this func will have dimensions drawn from one of the following: [ntimes, ncols, nrows, npix, ncomp]
    #   ntimes = number of cadences
    #   ncols, nrows = shape of tpf
    #   npix = ncols*nrows, is the length of the unraveled vectors
    #   ncomp = num of components in the polynomial

    # Error catching
    if not isinstance(tpf, lk.TessTargetPixelFile):
        raise ValueError("Input a TESS Target Pixel File")

    if (np.product(tpf.shape[1:]) < 100) | np.any(np.asarray(tpf.shape[1:]) < 6):
        raise ValueError("TPF too small. Use a bigger cut out.")

    # Grid for calculating polynomial
    R, C = np.mgrid[: tpf.shape[1], : tpf.shape[2]].astype(float)
    R -= tpf.shape[1] / 2
    C -= tpf.shape[2] / 2

    # nested b/c we run twice, once on each orbit
    def func(tpf):
        # Design matrix
        A = np.vstack(
            [
                R.ravel() ** idx * C.ravel() ** jdx
                for idx in range(polyorder + 1)
                for jdx in range(polyorder + 1)
            ]
        ).T

        # Median star image
        m = np.median(tpf.flux.value, axis=0)
        # Remove background from median star image
        mask = ~sigma_clip(m, sigma=3).mask.ravel()
        # plt.imshow(mask.reshape(m.shape))
        bkg0 = A.dot(
            np.linalg.solve(A[mask].T.dot(A[mask]), A[mask].T.dot(m.ravel()[mask]))
        ).reshape(m.shape)

        # m is the median frame
        m -= bkg0

        # Include in design matrix
        A = np.hstack([A, m.ravel()[:, None]])

        # Fit model to data, including a model for the stars in the last column
        f = np.vstack(tpf.flux.value.transpose([1, 2, 0]))
        ws = np.linalg.solve(A.T.dot(A), A.T.dot(f))
        # shape of ws is (num of times, num of components)
        # A . ws gives shape (npix, ntimes)

        # Build a model that is just the polynomial
        model = (
            (A[:, :-1].dot(ws[:-1]))
            .reshape((tpf.shape[1], tpf.shape[2], tpf.shape[0]))
            .transpose([2, 0, 1])
        )
        # model += bkg0
        return model

    # Break point for TESS orbit
    # currently selects where the biggest gap in cadences is
    # could cause problems in certain cases with lots of quality masking! Think about how to handle bit masking
    b = np.where(np.diff(tpf.cadenceno) == np.diff(tpf.cadenceno).max())[0][0] + 1

    # Calculate the model for each orbit, then join them
    model = np.vstack([func(tpf) for tpf in [tpf[:b], tpf[b:]]])

    return model



def roundup(x, pow=0):
    """
    Rounds up to the nearest power of ten, given by pow.
    """
    return int(math.ceil(x / (10 ** (pow)))) * (10 ** (pow))


def pixels_to_radius(cutout_size, pow=-2, round=True):
    """
    Given the side length of a cut out in pixels, converts it to the radius of a cone search in degrees, rounded up to the nearest `pow` power of 10. Use round to turn on/off the rounding up feature.
    """
    if round:
        return roundup(cutout_size / np.sqrt(2) * 21 / 3600, pow=pow)
    else:
        return cutout_size / np.sqrt(2) * 21 / 3600


def pix_to_arcsec(pix):
    return pix * 21.0


def pix_to_degrees(pix):
    return pix * 21.0 / 3600

def flux_to_mag(flux, reference_mag=20.44):
    """Converts a TESS magnitude to a flux in e-/s. The TESS reference magnitude is taken to be 20.44. If needed, the Kepler reference flux is 1.74e5 electrons/sec.
    
    Parameters
    ----------
    flux : float
        The total flux of the target on the CCD in electrons/sec.
    reference_mag: int
        The zeropoint reference magnitude for TESS. Typically 20.44 +/-0.05.
    reference_mag: float

    Returns
    -------
    Tmag: float
        TESS magnitude of the target.
    
    """
    # kepler_mag = 12 - 2.5 * np.log10(flux / reference_flux)
    mag = -2.5 * np.log10(flux) + reference_mag
    return mag


def mag_to_flux(Tmag, reference_mag=20.44):
    """Converts a TESS magnitude to a flux in e-/s. The TESS reference magnitude is taken to be 20.44. If needed, the Kepler reference flux is 1.74e5 electrons/sec.
    
    Parameters
    ----------
    Tmag: float
        TESS magnitude of the target.
    reference_mag: int
        The zeropoint reference magnitude for TESS. Typically 20.44 +/-0.05.

    Returns
    -------
    flux : float
        The total flux of the target on the CCD in electrons/sec.
    """
    # fkep = (10.0 ** (-0.4 * (mag - 12.0))) * 
    return 10 ** (-(Tmag - reference_mag)/2.5)

# Class for managing sparse 3D matrices, originally copied from pandorapsf to avoid a package dependency

class SparseWarp3D(sparse.coo_matrix):
    """Special class for working with stacks of sparse 3D images"""

    def __init__(self, data, row, col, imshape):
        if not np.all([row.ndim == 3, col.ndim == 3, data.ndim == 3]):
            raise ValueError("Pass a 3D array (nrow, ncol, nvecs)")
        self.nvecs = data.shape[-1]
        if not np.all(
            [
                row.shape[-1] == self.nvecs,
                col.shape[-1] == self.nvecs,
            ]
        ):
            raise ValueError("Must have the same 3rd dimension (nvecs).")
        self.subrow = row.astype(int)
        self.subcol = col.astype(int)
        self.subdepth = (
            np.arange(row.shape[-1], dtype=int)[None, None, :]
            * np.ones(row.shape, dtype=int)[:, :, None]
        )
        self.subdata = data
        self._kz = self.subdata != 0

        self.imshape = imshape
        self.subshape = row.shape
        self.cooshape = (np.prod([*self.imshape[:2]]), self.nvecs)
        self.coord = (0, 0)
        super().__init__(self.cooshape)
        index0 = (np.vstack(self.subrow)) * self.imshape[1] + (np.vstack(self.subcol))
        index1 = np.vstack(self.subdepth).ravel()
        self._index_no_offset = np.vstack([index0.ravel(), index1.ravel()])
        self._submask_no_offset = np.vstack(self._get_submask(offset=(0, 0))).ravel()
        self._subrow_v = deepcopy(np.vstack(self.subrow).ravel())
        self._subcol_v = deepcopy(np.vstack(self.subcol).ravel())
        self._subdata_v = deepcopy(np.vstack(deepcopy(self.subdata)).ravel())
        self._index1 = np.vstack(self.subdepth).ravel()

        self._set_data()

    def __add__(self, other):
        if isinstance(other, SparseWarp3D):
            data = deepcopy(self.subdata + other.subdata)
            if (
                (self.subcol != other.subcol)
                | (self.subrow != other.subrow)
                | (self.imshape != other.imshape)
                | (self.subshape != other.subshape)
            ):
                raise ValueError("Must have same base indicies.")
            return SparseWarp3D(
                data=data, row=self.subrow, col=self.subcol, imshape=self.imshape
            )
        else:
            return super(sparse.coo_matrix, self).__add__(other)

    def tocoo(self):
        return sparse.coo_matrix((self.data, (self.row, self.col)), shape=self.cooshape)

    def index(self, offset=(0, 0)):
        """Get the 2D positions of the data"""
        if offset == (0, 0):
            return self._index_no_offset
        index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
            self._subcol_v + offset[1]
        )
        #        index1 = np.vstack(self.subdepth).ravel()
        #        return np.vstack([index0.ravel(), index1.ravel()])
        # index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
        #     self._subcol_v * offset[1]
        # )
        return index0, self._index1
        # index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
        #     self._subcol_v * offset[1]
        # )
        # return index0, self._index1

    def _get_submask(self, offset=(0, 0)):
        # find where the data is within the array bounds
        kr = ((self.subrow + offset[0]) < self.imshape[0]) & (
            (self.subrow + offset[0]) >= 0
        )
        kc = ((self.subcol + offset[1]) < self.imshape[1]) & (
            (self.subcol + offset[1]) >= 0
        )
        return kr & kc & self._kz

    def _set_data(self, offset=(0, 0)):
        if offset == (0, 0):
            index0, index1 = self.index((0, 0))
            self.row, self.col = (
                index0[self._submask_no_offset],
                index1[self._submask_no_offset],
            )
            self.data = self._subdata_v[self._submask_no_offset]
        else:
            # find where the data is within the array bounds
            k = self._get_submask(offset=offset)
            k = np.vstack(k).ravel()
            new_row, new_col = self.index(offset=offset)
            self.row, self.col = new_row[k], new_col[k]
            self.data = self._subdata_v[k]
        self.coord = offset

    def __repr__(self):
        return (
            f"<{(*self.imshape, self.nvecs)} SparseWarp3D array of type {self.dtype}>"
        )

    def dot(self, other):
        if other.ndim == 1:
            other = other[:, None]
        nt = other.shape[1]
        return super().dot(other).reshape((*self.imshape, nt)).transpose([2, 0, 1])

    def reset(self):
        """Reset any translation back to the original data"""
        self._set_data(offset=(0, 0))
        self.coord = (0, 0)
        return

    def clear(self):
        """Clear data in the array"""
        self.data = np.asarray([])
        self.row = np.asarray([])
        self.col = np.asarray([])
        self.coord = (0, 0)
        return

    def translate(self, position: Tuple):
        """Translate the data in the array by `position` in (row, column)"""
        self.reset()
        # If translating to (0, 0), do nothing
        if position == (0, 0):
            return
        self.clear()
        self._set_data(position)
        return