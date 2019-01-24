Alternative treatment for "simple spot disentanglement".

Focus on one Bragg spot only, on one image.  It has intensity contributions from two colors, call them Blue and Red.  We will create a model to simulate the data, and then compare the simulation with experiment.  This will give us a residual to use for least squares (likelihood?) refinement of parameters.  We need to be clear what the unknown parameters are.

Unknown parameters:
$G$ an overall scale factor unique to the lattice that affords the best numerical correlation between simulation and experiment.

$I_{\text{Blue}}$ the structure factor intensity at the Blue energy.

$I_{\text{Red}}$ the structure factor intensity at the Red energy.

The simulated intensity of the integrated spot is 

$$ D_{\text{spot}} = \left{[} I_{\text{Blue}} k L_{\text{Blue}} P_{\text{Blue}} + I_{\text{Red}} k L_{\text{Red}} P_{\text{Red}}\right{]}$$

where 

L_{\text{Channel}} represents the number of photons impinging on sample from the energy channel, either Blue or Red.  Expressing it this way implicitly assumes that a special run of $\text{simtbx}$ is performed, for the purpose of producing the spot profile P_{\text{Channel}}, in which equal weight is given to each energy channel (not the normal way to run it).  

$k$ represents a "dummy" structure factor given in the special run of $\text{simtbx}$. The simple way to think about this dummy value is that it is the same for all Miller indices (at least those in a given resolution bin), and that it is independent of energy channel and which Bijvoet mate is chosen.

P_{\text{Channel}} is the summed intensities over all non-zero pixels in the simulated spot profile for a particular Miller index and a particular energy channel.

Clearly there are too many unknowns $G$, $I_{\text{Blue}}$, $I_{\text{Red}}$ to be treated by a single equation.  The idea is that there will be many residuals on each image, giving us the per-image $G$ factor, and there will be many observations of $I_{\text{Channel}}$ over the multi-image dataset, so we get a large overdetermined least-squares problem.
