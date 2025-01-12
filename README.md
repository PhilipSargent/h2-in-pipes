# h2-in-pipes
This work includes an Equation of state for natural gas (several varieties) and mixtures with hydrogen. We also have Moody plots for the calculation of the friction factor in rough pipes.

To cite this work, please click on "Cite this repository" on the right-hand menu.

This is research code. It is **not qualified for any design purpose whatsoever**. It contains bugs and has not been through any code review process.

We have a paper just accepted for publication (29/02/2024) in J.Intl Hydrogen Energy which documents and uses these calculations for the case where UK distribution gas pipes are repurposed to carry hydrogen.
If you would like a preprint of this paper, please just look at the file https://github.com/PhilipSargent/h2-in-pipes/blob/main/_SUBMIT3-BOTH-Hydrogen_pipe_friction.pdf in this repo.

The oxygen-firing effect on the dew point and thus the efficiency of condensing boilers is describe in an online article: https://philipsargent.wordpress.com/2024/03/01/oxygen-boilers-making-condensing-boilers-more-efficient/
![p2_h2_ratio](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/be00b721-4024-4d8e-ade0-a5a0f9c79bb7)

![peng_z_p](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/e7a252f0-f0b5-49a6-82f9-188b04dcb490)

![w2_h2_ratio_pt](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/35bf4e3c-8b86-4151-b591-4c3c261ae42b)

![moody_afzal](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/06769e5b-2bf3-4444-86ae-99622bd48139)

![peng_z](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/8c38150c-cf02-49b9-98d4-5269e28eb2ca)

![peng_bf](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/92a911eb-c0ea-41dc-91c2-0f80a38c8f22)
![visc-temp](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/9883a784-3bf2-4557-86d8-043a891bf011)

To deliver the same calorific value as hydrogen instead of natural gas, the hydrogen needs to be moving at least as 3x faster than the natural gas (int he same pipe), but the difference in flow regime, friction factor and compressibility factor Z all make a difference too:
![peng_v_ratio](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/6f19f581-b467-4fca-bb46-7fbbf56277f0)

![peng_ηη_H2](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/b835f9a9-5124-4ce6-a068-1f7d8aecf7d3)
![peng_ηη_NG](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/41b294d4-8631-4759-a906-3b7b8062a3c5)
![condse_η20](https://github.com/PhilipSargent/h2-in-pipes/assets/5623885/b0cfa7ee-120d-42db-9500-e1a9acc761a5)

We can use the models to calculate the linepack for natural gas and for hydrogen when flowing in the same pipe and delivering the same combustion energy (HHV):

![pipe_lpm.png](https://github.com/PhilipSargent/h2-in-pipes/blob/main/[pipe_lpm.png)

![pipe_lpm_ratio.png](https://github.com/PhilipSargent/h2-in-pipes/blob/main/pipe_lpm_ratio.png)
