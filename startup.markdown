# Processing of CXID9114

## Contents

* [Background](#background)
    * [Experimental overview](#overview)
    * [Analysis goals](#ana) 
* [Data processing](#dataproc)
    * [Integration of spots](#integrate)
        * [Disentanglement of spots](#disentangle)
        * [Simulated integrated intensities](#simulated)	
    * [Indexing](#indexing)
        * [Indexing assignment](#idx_ass) 
        * [Indexing refinement](#refinement)
* [Solving the MAD equations](#so_mad) 
* [Post refinement](#post)

<a name="background"></a>
## Background
<a name="overview"></a>

### Experiment overview
We index many snapshots (stills) of crystals as they are injected (using GDVN) into the sample-Xray interaction region. Prior to reaching the sample, each XFEL pulse is split into multiple energies (centered around 2 bright, sharp bands at 8944 and 9034 keV), and simultaneous diffraction is recorded for photons of all energies. Many of the measured reflections from the different wavelengths are therefore overlapped, which makes the data analysis challenging. 

<a name="ana"></a>
### Analysis goals
The goal of the data reduction is to determine wavelength dependent structure factors for each miller index $\lambda$, $$|F_\lambda|^2$$ and to show that there is a measurable contrast between Bijouvet pairs $|F^+_\lambda|^2$ and $|F^-_\lambda|^2$ where the $\pm$ refers to the miller indices $\pm\mathbf{H}$ (note that notation of $\mathbf{H}$ is suppressed in this document). In the MAD scattering measurement, when the heavy atom substructure within a crystal undergoes photo absorption, Friedel symmetry is broken leading to differences the aforementioned contrast. There exists a geometric relationship between $F+$ and $F-$  gives rise to a set of two equations with 3 unknown quantities (the Karle and Hendrickson equations), namely the structure factor magnitudes of the protein $F_p$ and heavy atom substructure $F_a$, as well as the angular difference between protein and heavy atom phase terms, $\alpha$. As the structure factors are wavelength-dependent near the absorption energy of the heavy atoms, measuring the $F+$ and $F-$ at multiple wavelengths then determines the solution of the equations (more wavelengths are needed for each heavy atom species) and $F_p$, $F_a$, and $\alpha$ may be solved for. In turn this information is used to solve the phase of the protein.

<a name="dataproc"></a>
## Data processing
<a name="integrate"></a>
### Integration of spots
The structure factors are measured many times across multiple snapshots. On a given snapshot, the strong spots are identified, and an indexing solutions is (hopefully) arrived at (see [indexing](#indexing)). Then, using the indexing solution, miller indices are assigned to each spot (see [indexing assignment](#idx_ass)). Optionally, indexing refinement is performed (see [indexing refinement](#refinement)). Now the question arises, how do we integrate the spots on each snapshot? Let's assume we integrate the spot in a standard way (detailed [below](#below)), and then we modify the question: **how do we distribute a spots intensity across its various wavelength contributions?**.

<a name="disentangle"></a>
### Simple spot disentanglement
Our underlying assumption here is the following relationship for integrated intensities on an area detector:

$$
\frac{I_\lambda}{I_{\text{tot}}} = \frac{S_\lambda}  {S_{ \text{tot}}}
$$

where $I_{\text{tot}} = \sum_\lambda I_\lambda$ is the contributions from the various wavelength channels and $S$ is simply a simulated version of $I$ (done with e.g. ```simtbx.nanoBragg```, see [simulated intensities](#simulated)). From our area detector data, we can calculate $I_{\text{tot}}$ of a reflection as follows:
<a name="below"></a>
$$ 
I_{\text{tot}} = \sum_{ij \,\in\, \text{bbox}} (D_{ij}-B_{ij}) \,M_{ij}
$$ 

where *bbox* refers to a local rectangular sub-image containing the reflection in question, $D_{ij}$ are the pixel values themselves (presumably corrected for polarization, solid angle and photon gain, etc), $M_{ij}$ is the strong spots mask specifying pixels containing crystal scatter, and $B_{ij}$ is the best-fit local background plane (or surface) evaluated using background-labeled pixels (i.e. pixels that are not labeled as strong-spots by $M_{ij}$, and also pixels that are not zingers).

Now, let $p$ represent the partiality of the spot, or the fraction of the full reflection that is measured on the camera. Also, let $c_\lambda$ represent the fraction of photons in the XFEL pulse at wavelength  $\lambda$.

We then have a formulation for estimating the wavelength dependent form factor:

$$
|F_\lambda|^2 = \frac{ (S_\lambda \,/\, S_{\text{tot}})\, I_{\text{tot}}}{ p \, c_\lambda}
$$

This is maybe cheating a bit because we choose the structure factors that go into the simulation, however maybe we can simply replace $(S_\lambda \,/\, S_{\text{tot}})$ with a scaling factor and refine this using the MAD equations (don't have a clear picture of how to do this yet).


<a name="simulated"></a>
### Simulated integrated intensities
In order to determine the $S_\lambda$ we decide which wavelengths are significant in the spectrometer measurement, and then we perform a simulation of the reflection for each wavelength. We then use a simple thresholding procedure for determining the strong spots in each of the $S_\lambda$. From the spot locations we can perform a simple integration (there is no background in the simulation) of the strong spots to get the overall contributions. 

I guess this is cheating a bit because we choose which structure factors go into the simulation, however we can also do a controlled version of this analysis where we choose different estimates for the structure factors, or rather assign scale factors to the structure factors used in the simulation, and then configure some kind of refinement

<a name="indexing"></a>
## Indexing
We use the two color grid search indexing technique. Crucially, we add Gaussian jitter to the magnitudes of the vectors in the basis vector grid search in order to accommodate variable crystal lattice dimensions from shot to shot. 

We eliminate all refinement that would be done by the stills indexer, and instead opt to refine at a later stage (though the jittering of the basis vector grid search is in-itself a shot-by-shot refinement). Actually, at this point we are still letting the original code choose the best orientation matrix, but even this step should probably be eliminated in favor of a simulation-based approach. 

Original only a few tens of thousands of hemisphere grid points were selected in the basis vector grid search, but now over 1.5 million vectors are beings tried per unit cell distance (in this case just a=79 Angstrom and c=38 Angstrom). For each basis vector of length a or c we add a Gaussian variance to its magnitude before checking how well it matches the periodicity in the data. The original basis vector search was also modified so that the points sampled on the unit hemisphere sphere now follow a standard spiral pattern as opposed to the previously implemented search space (rstbx/SimpleSamplerTool) which seemed to yield results very sensitive to the number of searching points, which was not desirable. Of these modifications, allowing the basis vectors to jitter was by far the most significant. 

<a name="idx_ass"></a>
###  Indexing assignment
In order to assign an $H$ to each measured reflection, use simtbx with the optimal crystal matrix to simulate the spots at each $\lambda$, then for each strong spot in the data, and for each wavelength we find the closest simulated spot. We use that simulated spots HKL (because we simulate onto a perfect geometry), and then we check whether the fractional miller indices $H_\text{frac}$ are each within 0.1 of the whole miler index $H$, that is $|h - h_\text{frac}| < 0.1$ and likewise for $k$ and $l$. If this condition is met, it is assumed that the wavelength can index the spot. Here is the code used to test indexability
<a name="refinement"></a>
### Indexing Refinement
In addition to the aforementioned basis vector grid search, here we jitter the orientation matrix determined by the indexing stage in order to obtain a better overlap with the data. This is accomplished by rocking the indexed crystal matrix about the $x$ and $y$ axis directions ($z$ is the beam direction) and using simulated diffraction patterns to computing a simple overlap between the simulation and the data, defined as

$$
\text{overlap} = \sum_{ij}M_{ij}\,S_{ij}
$$

where $M_{ij}$ is the strong spot mask from the data and $S_{ij}$ is the simulation. We also tried the metric

$$
\text{overlap} = \sum_{ij}M_{ij}\,M^S_{ij}
$$

is the strong spot mask from the simulation, but it is unclear at this point whether that helps or matters. We also used intensity to weight the overlap regions, but that also made little difference in preliminary tests. This stage is still being *refined* for lack of a better term. 

<a name="so_mad"></a>
### Solving the MAD equations

> These are Notes on my readings and thoughts into the matter, still need to read the Karle and MADLSQ papers

The standard MAD equations are

$$
|F_\lambda^\pm|^2 = F_p^2 + a_\lambda F_a^2 + b_\lambda F_p F_a \cos \alpha  \pm c_\lambda F_p F_a \sin \alpha   
$$

where the structure and wavelength dependence are separated, and the terms $a_\lambda$, $b_\lambda$, and $c_\lambda$ are the atomic scattering factors of the heavy atom species substructure assumed to be measured before the experiment for the crystals under investigation. 

#### Solve using calculus

Rewrite the equation above as

$$
|F_\lambda^\pm|^2 = x + a_\lambda y + b_\lambda u \pm c_\lambda v  
$$

and note that 

$$
u^2 + v^2 = xy
$$

which provides a constraint. Then Lagrange multipliers method can be used to solve the five equations in five unknowns

$$
\nabla_{x,y,u,v} |F^+|^2 = \gamma\, \nabla_{x,y,u,v}G
$$
$$
G = u^2 + v^2 - xy = 0
$$

where $\gamma$ is the Lagrange multiplier.

> Then what ? 

#### Solve using algebra
What follows *might* be dumb, but I write it here for discussion and I think it solves the equations albeit in a rather ugly way. 

Looking at these equations, I am eager to eliminate the sin and cos terms in favor of a single equation with two unknowns (this seems to be what the Lagrangian multiplier constraint does implicitly). 

I would just add and subtract these two equations to form 2 new equations

$$
|F_\lambda^+|^2 + |F_\lambda^-|^2 = 2F_p^2 + 2a_\lambda F_a^2 + 2 b_\lambda F_p F_a \cos \alpha
$$

and then 

$$
|F_\lambda^+|^2 - |F_\lambda^-|^2 =  2 c_\lambda F_p F_a \sin \alpha
$$

Now we can solve for the $\cos$ and $\sin$ terms and then square them and add them to obtain a single equation. First lets make definitions 
$$u \equiv |F^+|^2 + |F^-|^2$$
$$v \equiv |F^+|^2 - |F^-|^2$$
then lets re-scale the $a$, $b$, $c$ terms by 2, and rename $x\equiv F_p$ and $y\equiv F_a$ . Then we get

$$u = 2x^2 + a\, y^2 +  b\, xy\cos\alpha$$
$$v = c\, xy\sin\alpha$$
Rearranging we get 

$$xy \cos\alpha = (u - 2x^2 - ay^2) / b $$

$$xy \sin\alpha = v/c$$

and hence we arrive at the single equation in two unknowns
$$
(xy)^2 = \left (\frac{u-2x^2-ay^2}{b}\right )^2 + v^2 / c^2
$$

Now, we have this equation once for each wavelength $\lambda$, as the terms $u,v,a,b,c$ are all $\lambda$-dependent, and therefore by measuring at two wavelengths we can solve for $x$ and $y$.. I think there is an exact set of 4 solutions in the case of 2 wavelengths (according to wolfram alpha).
