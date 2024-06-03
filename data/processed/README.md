# PLATO Targets Classification

This repository contains the full list of PLATO targets, classified into their Galactic Component membership. The basis for the catalogue is the all-sky PLATO input catalog (asPIC) by Montalto2021, crossmatched with Gaia DR3.

## Files

There are four files included in this repository:

1. **targets_classified**:
   - Description: The entire sample, classified into thin disk, thick disk candidate, thick disk, halo candidate, and halo.
   - Note : Classification is only performed if relative uncertainties no greater than 20% in the following columns: "ra", "dec", "pmra", "pmdec", "parallax", and "radial_velocity". The remaining columns are filled with NaN.
   - Format: 2,675,538 rows, 61 columns

2. **LOPS2**:
   - Description: A subselection of targets_classified in the LOPS2 field.
   - Galactic Coordinates: l=255.9375°, b=-24.62432°
   - Additional Column: "n_cameras" specifying the number of PLATO cameras that will observe the target.
   - Format: 169,438 rows, 62 columns

3. **LOPN1**:
   - Description: A subselection of targets_classified in the LOPN1 field.
   - Galactic Coordinates: l=81.56250°, b=24.62432°
   - Format: 173,735 rows, 62 columns

4. **special_target_list**:
   - Description: A selection of 51 targets, kinematically classified as halo stars, within the high-SNR P1 PLATO sample.
   - Additional Column: "n_cameras" specifying the number of PLATO cameras that will observe the target, and "Field" specifying if target belongs to LOPS2 or LOPN1 field
   - Format: 51 rows, 63 columns

## Classification Criteria

We classify only those stars into Galactic components that have relatively certain kinematic parameters. Specifically, we require no more than 20% relative uncertainty in the following columns: "ra", "dec", "pmra", "pmdec", "parallax", "radial_velocity". This results in 2,283,538 out of the total 2,675,539 stars having a component classification. The other columns will have NaNs.

## Analysis Conditions

For the analysis in our paper, we applied the following additional conditions:

- Population.notnull()
- Stellar type = FGK
- Radius > 0
- Mass > 0
- Teff > 0
- Logg.notnull()
- [Fe/H] < 1

Applying these conditions results in 1,999,021 entries remaining.

## Columns

The columns in the dataset are as follows:

- **gaiaID_DR2**: Gaia DR2 ID (from asPIC catalog, Montalto2021)
- **gaiaID_DR3**: Gaia DR3 ID (matched to DR2 catalog, based on angular distance)
- **parallax**: Parallax of target (mas), from Gaia DR3
- **e_parallax**: Error on parallax
- **ra**: Right Ascension (deg) from Gaia DR3
- **e_ra**: Error on Right Ascension
- **dec**: Declination (deg) from Gaia DR3
- **e_dec**: Error on declination
- **pmra**: Proper motion in right ascension direction (mas/yr), from Gaia DR3
- **e_pmra**: Error on proper motion in right ascension direction
- **pmdec**: Proper motion in declination direction (mas/yr), from Gaia DR3
- **e_pmdec**: Error on proper motion in declination direction
- **radial_velocity**: Radial velocity (km/s), from Gaia DR3
- **e_radial_velocity**: Radial velocity error
- **[alpha/Fe]**: Alpha elemental abundance from Gaia RV spectra, alphafe_gspspec in Gaia DR3
- **e_[alpha/Fe]_lower**: Lower error on alpha element abundance
- **e_[alpha/Fe]_upper**: Upper error on alpha element abundance
- **[Fe/H]**: Metallicity estimate, either from RVS spectra (mh_gsspec in Gaia DR3), BP/RP spectra (mh_gspphot in Gaia DR3) or XGBOOST (Andrae2023)
- **[Fe/H]_source**: Source of the metallicity estimate
- **e_[Fe/H]_lower**: Lower error on the metallicity estimate (NaN in case of Andrae2023)
- **e_[Fe/H]_upper**: Upper error on the metallicity estimate (NaN in case of Andrae2023)
- **logg**: log g estimate, either from RVS spectra (logg_gsspec in Gaia DR3), BP/RP spectra (logg_gspphot in Gaia DR3) or XGBOOST (Andrae2023)
- **logg_source**: Source of the log g estimate
- **e_logg_lower**: Lower error on the log g estimate (NaN in case of Andrae2023)
- **e_logg_upper**: Upper error on the log g estimate (NaN in case of Andrae2023)
- **[Fe/H]_apogee**: Metallicity estimate from APOGEE-DR17 (not available for all targets)
- **e_[Fe/H]_apogee**: Error on APOGEE metallicity estimate
- **[alpha/M]_apogee**: Alpha abundance estimate from APOGEE-DR17 (not available for all targets)
- **e_[alpha/M]_apogee**: Error on APOGEE alpha estimate
- **logg_apogee**: log g estimate from APOGEE-DR17 (not available for all targets)
- **e_logg_apogee**: Error on APOGEE log g estimate 
- **[Fe/H]_galah**: Metallicity estimate from GALAH DR3 (not available for all targets)
- **e_[Fe/H]_galah**: Error on GALAH metallicity
- **[alpha/Fe]_galah**: Alpha abundance estimate from GALAH DR3 (not available for all targets)
- **e_[alpha/Fe]_galah**: Error on GALAH alpha abundance
- **logg_galah**: log g estimate from GALAH DR3 (not available for all targets)
- **e_logg_galah**: Error on GALAH log g
- **GLON**: Galactic longitude (deg)
- **GLAT**: Galactic latitude (deg)
- **U**: Heliocentric velocity in the direction of the Galactic center (km/s), calculated from ra, dec, parallax, pmra, pmdec, radial_velocity using galpy, only available if Population could be determined (see Classification Criteria)
- **V**: Heliocentric velocity in the direction of the Galactic rotation (km/s), calculated from ra, dec, parallax, pmra, pmdec, radial_velocity using galpy, only available if Population could be determined (see Classification Criteria)
- **W**: Heliocentric velocity in the direction of the North Galactic Pole (km/s), calculated from ra, dec, parallax, pmra, pmdec, radial_velocity using galpy, only available if Population could be determined (see Classification Criteria)
- **UW**: Total non-circular velocity UW = sqrt(U^2 + W^2) (km/s), only available if Population could be determined (see Classification Criteria)
- **R**: Distance from the Galactic center in the Galactic plane (kpc), only available if Population could be determined (see Classification Criteria)
- **Z**: Distance from the Galactic plane (kpc), only available if Population could be determined (see Classification Criteria)
- **gaiaV**: V-band magnitude (from asPIC catalog, Montalto2021)
- **e_gaiaV**: Error on V-band magnitude 
- **Gmag**: Gaia G magnitude (from asPIC catalog, Montalto2021)
- **e_Gmag**: Error on G magnitude
- **Radius**: Radius of star (solar radii) (from asPIC catalog, Montalto2021)
- **e_Radius**: Error on radius
- **Mass**: Mass of star (solar masses) (from asPIC catalog, Montalto2021)
- **e_Mass**: Error on mass
- **Teff**: Effective temperature of star (K) (from asPIC catalog, Montalto2021)
- **e_Teff**: Error on effective temperature
- **Stellar Type**: Classification into M or FGK type, based on asPIC catalog (Montalto2021)
- **u1**: The first quadratic limb-darkening parameter, based on grid by Morello2022
- **u2**: The second quadratic limb-darkening parameter, based on grid by Morello2022
- **TD/D**: Probability ratio thick disk/thin disk (see Section 2.2. in our paper), only available if Population could be determined (see Classification Criteria)
- **TD/H**: Probability ratio thick disk/halo (see Section 2.2 in our paper), only available if Population could be determined (see Classification Criteria)
- **Population**: Galactic component classification (Thin Disk, Thick Disk, Halo, Thick Disk Candidate, Halo Candidate), see paper for definitions, only available if Population could be determined (see Classification Criteria)

### Additional Columns for LOPS2, LOPN1, and Special Target List

- **n_cameras**: Number of PLATO cameras observing the target

### Additional Column for Special Target List

- **field**: Flag for LOPS2 or LOPN1 fields
