 
### Background

We index many snapshots (stills) of crystals as they are injected (using GDVN) into the sample-Xray interaction region. Prios to reaching the sample, each XFEL pulse is split into multiple energies (centered around 2 bright, sharp bands at 8944 and 9034 keV), and simultaneous diffraction is recorded for photons of all energies. Many of the measured reflections from the different wavelengths are therefore overlapped, which makes the data anslysis challenging. 

### Data reduction
The goal of the data reduction is to determine wavelength dependent structure factors for each miller index $\lambda$, $$|F_\lambda|^2$$ and to show that there is a measurable contrast between Bijouvet pairs $|F^+_\lambda|^2$ and $|F^-_\lambda|^2$ where the $\pm$ refers to the miller indices $\pm\mathbf{H}$ (note that notation of $\mathbf{H}$ is suppressed in this document). In the MAD scattering measurement, when the heavy atom substructure within a crystal undergoes photo absorption, Friedel syymetry is broken leading to differences the aforementioned contrast. There exists a geometric relationship between $F+$ and $F-$  gives rise to a set of two equations with 3 unknown quantities (the Karle and Hendrickson equations), namely the structure factor magnitudes of the protein $F_p$ and heavy atom substructure $F_a$, as well as the anglular difference between protein and heavy atom phase terms, $\alpha$. As the structure factors are wavelength-dependent near the absorption energy of the heavy atoms, measuring the $F+$ and $F-$ at multiple wavelengths then determines the solution of the equations (more wavelengths are needed for each heavy atom species) and $F_p$, $F_A$, and $\alpha$ may be solved for. In turn this information is used to solve the phase of the protein.

### Data processing
The structure factors are measured many times across multiple snapshots. On a given snapshot, the strong spots are identified, and an indexing solutions is (hopefully) arrived at [see [indexing](#indexing)]. Then, using the indexing solution, miller indices are assigned to each spot [see indexing assignment]. Optionally, indexing refinement is performed [see indexing refinement]. Now the question arises, how do we integrate the spots on each snapshot? Lets assume we integrate the spot in a standard way (detailed below), and then we modify the question: **how do we distribute a spots intensity across its various wavelength contributions?**.

#### Simple spot disentanglement
Our underlying assumption here is the following relationship for integrated intensities on an area detector:

$$
\frac{I_\lambda}{I_{\text{tot}}} = \frac{S_\lambda}  {S_{ \text{tot}}}
$$

where $I_{\text{tot}} = \sum_\lambda I_\lambda$ is the contributions from the various wavelength channels and $S$ is simply a simulated version of $I$ (done with e.g. ```simtbx.nanoBragg```, see [simulated intensities](#simulated)). From our area detector data, we can calculate $I_{\text{tot}}$ of a reflection as follows:

$$ 
I_{\text{tot}} = \sum_{ij \,\in\, \text{bbox}} (D_{ij}-B_{ij}) \,M_{ij}
$$ 

where *bbox* refers to a local rectangular sub-image containing the reflection in question, $D_{ij}$ are the pixel values themselves (presumably corrected for polarization, solid angle and photon gain, etc), $M_{ij}$ is the strong spots mask specifying pixels containing crystal scatter, and $B_{ij}$ is the best-fit local background plane (or surface) evaluated using background-labeled pixels (i.e. pixels that are not labeled as strong-spots by $M_{ij}$, and also pixels that are not zingers).

Now, let $p$ represent the partiality of the spot, or the fraction of the full reflection that is measured on the camera. Also, let $c_\lambda$ represent the fraction of photons in the XFEL pulse at wavelength  $\lambda$.

We then have a formulation for estimating the wavelength dependent form factor:

$$
|F_\lambda|^2 = \frac{ (S_\lambda \,/\, S_{\text{tot}})\, I_{\text{tot}}}{ p \, c_\lambda}
$$

<a name="simulated"></a>
### Simulated integrated intensities
In order to determine the $S_\lambda$ we decide which wavelengths are significant in the spectrometer measurement, and then we perform a simulation of the reflection for each wavelength. We then use a simple thresholding procedure for determining the strong spots in each of the $S_\lambda$. From the spot locations we can perform a simple integration (there is no background in the simulation) of the strong spots to get the overal contributions. 

I guess this is cheating a bit because we choose which structure factors go into the data, however we can also do a controlled version of this analysis where we choose different estimates for the structure factors, or rather assign scale factors to the structure factors used in the simulation, and then configure some kind of refinement (dont have a clear picture as to how to do this yet).

<a name="indexing"></a>
### Indexing
We use the two color grid search indexing technique. Crucially, we add jitter to the basis vector grid search to accomodate for varying crystal lattice dimensions from shot to shot. We eliminate all refinement that would be done by the stills indexer, and instead opt to refine at a later stage, though the jittering of the basis vector grid search is initself a shot-by-shot refinement. The original basis vector search was also modifed so that the points sampled on the unit sphere followed a spiral pattern as opposed to the previously implemented search space. This lead to less sensitivity in the number of chosen grid points.


###  Indexing assignment




