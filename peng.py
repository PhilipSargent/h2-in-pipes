import functools 
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import sys

"""This code written Philip Sargent, starting in December 2023, by  to support a paper on the replacement of natural
gas in the UK distribution grid with hydrogen.

For general information, commercial natural gas typically contains 85 to 90 percent methane, with the remainder mainly nitrogen and ethane, and has a calorific value of approximately 38 megajoules (MJ) per cubic metre

Wobbe Number (WN) (i) ≤51.41 MJ/m³, and  (ii)  ≥47.20 MJ/m³
https://www.nationalgas.com/data-and-operations/quality
≥46.5MJ/m3 after 6 April 2025 https://www.hse.gov.uk/gas/gas-safety-management-regulation-changes.htm

UNITS: bar, K, litres
"""
# This algorithm does NOT deal with the temperature dependence of alpha properly. 
# The code should be rearranged to calculate alpha for each point ont he plot for each gas.

# Peng-Robinson Equation of State constants for Hydrogen and Methane
# Omega is the acentric factor is a measure of the non-sphericity of molecules; 
# a higher acentric factor indicates greater deviation from spherical shape
# PR constants data from ..aargh lost it.

R = 0.083144626  # l.bar/(mol.K)  SI after 2019 redefinition of Avagadro and Boltzmann constants
# 1 bar is today defined as 100,000 Pa not 1atm

Atm = 1.01325 # bar 
T273 = 273.15 # K


# "In this Schedule, the reference conditions are 15C and 1.01325 bar"
# UK law for legal Wobbe limits and their calculation.
# https://www.legislation.gov.uk/uksi/2023/284/made

# L M N for H2 from https://www.researchgate.net/figure/Coefficients-for-the-Twu-alpha-function_tbl3_306073867
# 'L': 0.7189,'M': 2.5411,'N': 10.2,
# for cryogenic vapour pressure. This FAILS for room temperature work, producing infinities. Use omega instead.
# possibly re-visit using the data from Jaubert et al. on H2 + n-butane.

# Hc is standard enthalpy of combustion at 298K, i.e. HHV,  in MJ/mol (negative sign not used). 
#     Data from Wikipedia & https://www.engineeringtoolbox.com/standard-heat-of-combustion-energy-content-d_1987.html
# Wb is Wobbe index: MJ/m³
# RD is relative density (air is  1)
# Vs is viscosity and temp. f measurement as tuple (microPa.s, K, exponent(pressure))
# All viscosities from marcia l. huber and allan h. harvey,
#https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=907539

# Viscosity temp ratameters calibrated by log-log fit to https://myengineeringtools.com/Data_Diagrams/Viscosity_Gas.html
# using aux. program visc-temp.py in this repo.

gas_data = {
    'H2': {'Tc': 33.2, 'Pc': 13.0, 'omega': -0.22, 'Mw':2.015, 'Vs': (8.9,300, 0.692), 'Hc':0.286},
    'CH4': {'Tc': 190.56, 'Pc': 45.99, 'omega': 0.01142, 'Mw': 16.04246, 'Vs': (11.1,300, 1.03), 'Hc':0.89058}, # Hc from ISO_6976
    'C2H6': {'Tc': 305.32, 'Pc': 48.72, 'omega': 0.099, 'Mw': 30.07, 'Vs': (9.4,300, 0.87), 'Hc':@misc{Sargents_github,
author = {Sargent, Philip and Sargent, Michael},
doi = {10.5281/zenodo.10368611},
month = dec,
title = {{Hydrogen in pipes}},
url = {https://github.com/PhilipSargent/h2-in-pipes},
version = {0.1.1},
year = {2023}
}
@misc{WikipediaNTS,
   author = {Wikipedia},
   title = {National Transmission System},
   url = {https://en.wikipedia.org/wiki/National_Transmission_System},
   year = {2023},
}

@misc{IGEM-G-13,
   author = {Institution of Gas Engineers and Managers},
   isbn = {9781739298111},
   title = {DOMESTIC SUPPLY CAPACITY AND OPERATING PRESSURE AT THE OUTLET OF THE METER},
   url = {https://www.igem.org.uk/resource/igem-g-13-domestic-supply-capacity-and-operating -pressure-at-the-outlet-of-the-meter.html},
   year = {2023},
}

@article{Schouten2004,
   author = {Jan A Schouten and Jan P J Michels and Renee Janssen-van Rosmalen},
   journal = {International Journal of Hydrogen Energy},
   pages = {1173-1180},
   title = {Effect of H2-injection on the thermodynamic and transportation properties of natural gas},
   volume = {29},
   url = {https://api.semanticscholar.org/CorpusID:98223314},
   year = {2004},
}
@article{Pina-Martinez2022,
   author = {Andrés Piña-Martinez and Romain Privat and Jean Noël Jaubert},
   doi = {10.1002/aic.17518},
   issn = {15475905},
   issue = {2},
   journal = {AIChE Journal},
   month = {2},
   publisher = {John Wiley and Sons Inc},
   title = {Use of 300,000 pseudo-experimental data over 1800 pure fluids to assess the performance of four cubic equations of state: SRK, PR, tc-RK, and tc-PR},
   volume = {68},
   year = {2022},
}
@misc{BEIS-Kiwa2021,
   author = {BEIS},
   title = {Review of the methodology for FGHRS in SAP Final report},
   url = {https://assets.publishing.service.gov.uk/media/614b218ee90e077a2db2e793/review-methodology-fghrs-sap.pdf},
   year = {2021},
}

@article{bossel2006,
  title={Does a Hydrogen Economy Make Sense?},
  author="Bossel, U.",
  url={http://www.industrializedcyclist.com/ulf%20bossel.pdf},
  year={2006},
  JOURNAL = {Proceedings of the IEEE},
  VOLUME  = {94},
  NUMBER  = {10},
  doi = {10.1109/JPROC.2006.883715},
  PAGES   = {1826--1837},
  publisher={IEEE}
}
@article{dodds2013,
  title="Conversion of the UK gas system to transport hydrogen",
  author="P.E. Dodds  and S. Demoullin ",
  url={http://dx.doi.org/10.1016/j.ijhydene.2013.03.070},
  year={2013},
  JOURNAL = {International Journal of Hydrogen Energy},
  VOLUME  = {38},
  NUMBER  = {18},
  doi = {10.1016/j.ijhydene.2013.03.070},
  PAGES   = {7189–-7200},
  publisher={Elsevier}
}
@article{Galyas2023,
   author = {Anna Bella Galyas and Laszlo Kis and Laszlo Tihanyi and Istvan Szunyog and Marianna Vadaszi and Adam Koncz},
   doi = {10.1016/j.ijhydene.2022.12.198},
   issn = {03603199},
   issue = {39},
   journal = {International Journal of Hydrogen Energy},
   keywords = {Energy capacity of pipelines,Hydrogen energy storage,Hydrogen-natural gas mixture,Pipeline transportation of hydrogen},
   month = {5},
   pages = {14795-14807},
   publisher = {Elsevier Ltd},
   title = {Effect of hydrogen blending on the energy capacity of natural gas transmission networks},
   volume = {48},
   year = {2023},
}
@misc{Friend1989,
   author = {Daniel G Friend and James F Ely and Hepburn Ingham},
   institution = {National Institute of Standards and Technology Technical Note 1325 Natl. Inst. Stand. Technol., Tech. Note 1325, 490 pages (Apr. 1989) CODEN:NTNOEF},
   note = {compressibility of methane},
   title = {Tables for the Thermophysical Properties of Methane},
   url = {https://www.govinfo.gov/content/pkg/GOVPUB-C13-c19f48b42b2c70c3422cda30d301b8b2/pdf/GOVPUB-C13-c19f48b42b2c70c3422cda30d301b8b2.pdf},
   year = {1989},
}
@article{Pina-Martinez2019,
   author = {Andrés Pina-Martinez and Romain Privat and Jean-Noël Jaubert and Ding-Yu Peng},
   doi = {10.1016/j.fluid.2018.12.007},
   issn = {03783812},
   journal = {Fluid Phase Equilibria},
   month = {4},
   pages = {264-269},
   title = {Updated versions of the generalized Soave α-function suitable for the Redlich-Kwong and Peng-Robinson equations of state},
   volume = {485},
   year = {2019},
}

@article{Lozana2022,
   author = {Daniel Lozano-Martín and Alejandro Moreau and César R Chamorro},
   doi = {https://doi.org/10.1016/j.renene.2022.08.096},
   issn = {0960-1481},
   journal = {Renewable Energy},
   keywords = {Equations of state,Hydrogen mixtures,Thermophysical properties},
   pages = {1398-1429},
   title = {Thermophysical properties of hydrogen mixtures relevant for the development of the hydrogen economy: Review of available experimental data and thermodynamic models},
   volume = {198},
   url = {https://www.sciencedirect.com/science/article/pii/S096014812201271X},
   year = {2022},
}
@misc{DESNZ2023a,
   author = {DESNZ},
   city = {Department for Energy Security and Net Zero},
   institution = { Department for Energy Security and Net Zero},
   month = {9},
   title = {ECUK 2023: End uses data tables (Excel)},
   url = {https://www.gov.uk/government/statistics/energy-consumption\-in\-the\-uk-2023},
   year = {2023},
}
@article{Witek2022,
   abstract = {The aim of this work is to examine the impact of the hydrogen blended natural gas on the linepack energy under emergency scenarios of the pipeline operation. Production of hydrogen from renewable energy sources through electrolysis and subsequently injecting it into the natural gas network, gives flexibility in power grid regulation and the energy storage. In this context, knowledge about the hydrogen percentage content, which can safely effect on materials in a long time steel pipeline service during transport of the hydrogen-natural gas mixture, is essential for operators of a transmission network. This paper first reviews the allowable content of hydrogen that can be blended with natural gas in existing pipeline systems, and then investigates the impact on linepack energy with both startup and shutdown of the compressors scenarios. In the latter case, an unsteady gas flow model is used. To avoid spurious oscillations in the solution domain, a flux limiter is applied for the numerical approximation. The GERG-2008 equation of state is used to calculate the physical properties. For the case study, a tree-topological high pressure gas network, which have been in-service for many years, is selected. The outcomes are valuable for pipeline operators to assess the security of supply.},
   author = {Maciej Witek and Ferdinand Uilhoorn},
   doi = {10.24425/ather.2022.143174},
   issn = {20836023},
   issue = {3},
   journal = {Archives of Thermodynamics},
   keywords = {Existing steel pipeline,Hydrogen blended natural gas,Hydrogen percentage content,Linepack energy},
   pages = {111-124},
   publisher = {Polska Akademia Nauk},
   title = {Impact of hydrogen blended natural gas on linepack energy for existing high pressure pipelines},
   volume = {43},
   year = {2022},
}

@article{Rosenow2024,
   abstract = {Graphical abstract Highlights d Comprehensive meta-review of 54 independent studies on heating with hydrogen d No studies support heating with hydrogen at scale d Evidence suggests heating with hydrogen is less efficient and more costly Authors Jan Rosenow Correspondence jan.rosenow@ouce.ox.a.uk In brief The scientific evidence does not support the widespread use of hydrogen for heating buildings. This is because it is less efficient, more costly, and more environmentally harmful than alternatives such as heat pumps and district heating. SUMMARY In the context of achieving net zero climate targets, heating poses a significant decarbonization challenge, with buildings contributing substantially to global energy consumption and carbon emissions. While enhancing energy efficiency in building fabric can reduce emissions, complete elimination is not feasible while relying on fossil-fuel-based heating systems. Hydrogen has been suggested for decarbonizing buildings in recent years as a potential solution for replacing fossil-fuel heating. This paper carries out a meta-review of 54 independent studies to assess the scientific evidence for using hydrogen for heating buildings. The analysis concludes that the scientific evidence does not support a major role for hydrogen in cost-optimal decarbonization pathways being associated with higher energy system and consumer costs. Electrification and district heating are deemed preferable due to higher efficiency and lower costs in the majority of analyzed studies.},
   author = {Jan Rosenow},
   doi = {10.1016/j.crsus.2023.100010},
   journal = {Cell Reports Sustainability},
   keywords = {consumer costs,electrification,heat pumps,heating,hydrogen,modeling,system costs},
   pages = {100010},
   title = {A meta-review of 54 studies on hydrogen heating},
   year = {2023},
}
@article{Moody1944,
   author = {L.F. Moody},
   issue = {8},
   journal = {Transactions of the ASME},
   pages = {671-684},
   title = {Friction Factors for Pipe Flow},
   volume = {66},
   year = {1944},
}
@misc{DESNZ2023b,
   author = {DESNZ},
   institution = {Department for Energy Security and Net-Zero},
   month = {12},
   title = {Lab testing - boiler cycling},
   url = {https://assets.publishing.service.gov.uk/media/65785a000467eb000d55f5ce/hem-val-05-lab-testing-boiler-cycling.pdf},
   year = {2023},
}


@misc{GASTEC2009,
   author = {Georgina Orr and Tom Lelyveld and BurtonSimon},
   institution = {GASTEC at CRE Ltd  for Energy Saving Trust },
   month = {6},
   title = {Final Report: In-situ monitoring of efficiencies of condensing boilers and use of secondary heating},
   url = {https://assets.publishing.service.gov.uk/media/5a75149be5274a3cb28697f7/In-situ_monitoring_of_condensing_boilers_final_report.pdf},
   year = {2009},
}

@misc{CleaverBooks2016,
   author = {CleaverBooks},
   journal = {web page document},
   month = {9},
   title = {The impact of excess air on efficiency},
   url = {https://www.watmfg.com/watmfg23082016/wp-content/uploads/2016/09/Excess-Air-and-Boiler-Efficiency.pdf},
   year = {2016},
}

@misc{pipesize2022,
   author = {DIYwiki},
   url={https://wiki.diyfaq.org.uk/index.php/Gas_pipe_sizing},
   month = {1},
   title = {Gas Pipe Sizing},
   year = {2022},
}

@misc{h2toolsz,
   author = {h2tools.org},
   title = {Compressibility of Hydrogen},
   url = {https://h2tools.org/hyarc/hydrogen-data/hydrogen-compressibility-different-temperatures-and-pressures},
   year = {2023},
}

@article{Plascencia2020,
   author = {Gabriel Plascencia and Lamberto Díaz–Damacillo and Minerva Robles-Agudo},
   doi = {10.1007/s42452-020-1938-6},
   issn = {25233971},
   issue = {2},
   journal = {SN (Springer Nature) Applied Sciences},
   keywords = {Artificial intelligence,Colebrook equation,Friction factor,Lambert W function,Moody chart},
   month = {2},
   publisher = {Springer Nature},
   title = {On the estimation of the friction factor: a review of recent approaches},
   volume = {2},
   year = {2020},
}
@misc{NationalGrid2017,
   author = {National Grid},
   title = {The Future of Gas Progress Report},
   year = {2017},
}

@article{Praks2020,
   author = {Pavel Praks and Dejan Brkić},
   doi = {10.23967/j.rimni.2020.09.001},
   issue = {3},
   journal = {Rev. int. métodos numér. cálc. diseño ing.},
   month = {9},
   pages = {41},
   title = {Review of New Flow Friction Equations: Constructing Colebrook’s Explicit Correlations Accurately},
   volume = {36},
   url = {https://api.semanticscholar.org/CorpusID:243543616 https://www.scipedia.com/public/Praks_Brkic_2020a},
   year = {2020},
}


@article{Samsatli2019,
   author = {Sheila Samsatli and Nouri J. Samsatli},
   doi = {10.1016/j.apenergy.2018.09.159},
   issn = {03062619},
   journal = {Applied Energy},
   keywords = {Design, planning and operation,Hydrogen for heat,Hydrogen supply chain,Integrated multi-vector networks,MILP,Value Web Model,Value chain optimisation},
   month = {1},
   pages = {854-893},
   publisher = {Elsevier Ltd},
   title = {The role of renewable hydrogen and inter-seasonal storage in decarbonising heat – Comprehensive optimisation of future renewable energy value chains},
   volume = {233-234},
   year = {2019},
}

@article{Wit2018,
   abstract = {Developing an economy based on a reduction in the use of fossil fuels in power generation and transport leads to an increased interest in hydrogen as the energy carrier of the future. Pipeline transmission appears to be the most economical means of transporting large quantities of hydrogen over great distances. However, before hydrogen can be widely used, a new network of pipelines will have to be constructed to ensure its transport. An alternative to the rather costly investment in the new infrastructure could be a utilization of the existing network of gas pipelines by adding hydrogen to natural gas and transporting the mixture. The new solution should be analysed regarding issues related to compression and the transport process itself, but consideration should also be given to the problem of the consequences of a gas pipeline failure. The paper presents the results of a comprehensive analysis of the process of compression and pipeline transport of the natural gas/hydrogen mixture with safety issues.},
   author = {Andrzej Witkowski and Andrzej Rusin and Mirosław Majkut and Katarzyna Stolecka},
   doi = {https://doi.org/10.1016/j.ijpvp.2018.08.002},
   issn = {0308-0161},
   journal = {International Journal of Pressure Vessels and Piping},
   keywords = {Compression,Natural gas/hydrogen mixture,Pipeline transport,Risk},
   pages = {24-34},
   title = {Analysis of compression and transport of the methane/hydrogen mixture in existing natural gas pipelines},
   volume = {166},
   url = {https://www.sciencedirect.com/science/article/pii/S0308016118301698},
   year = {2018},
}
@article{Offor2016,
   abstract = {The implicit Colebrook equation has been the standard for estimating pipe friction factor in a fully developed turbulent regime. Several alternative explicit models to the Colebrook equation have been proposed. To date, most of the accurate explicit models have been those with three logarithmic functions, but they require more computational time than the Colebrook equation. In this study, a new explicit non-linear regression model which has only two logarithmic functions is developed. The new model, when compared with the existing extremely accurate models, gives rise to the least average and maximum relative errors of 0.0025% and 0.0664%, respectively. Moreover, it requires far less computational time than the Colebrook equation. It is therefore concluded that the new explicit model provides a good trade-off between accuracy and relative computational efficiency for pipe friction factor estimation in the fully developed turbulent flow regime.},
   author = {Uchechukwu Herbert Offor and Sunday Boladale Alabi},
   doi = {10.4236/aces.2016.63024},
   issn = {2160-0392},
   issue = {03},
   journal = {Advances in Chemical Engineering and Science},
   pages = {237-245},
   publisher = {Scientific Research Publishing, Inc.},
   title = {An Accurate and Computationally Efficient Explicit Friction Factor Model},
   volume = {06},
   year = {2016},
}
@article{Colebrook1939,
   author = {C. F. Colebrook},
   doi = {10.1680/ijoti.1939.13150},
   issn = {0368-2455},
   issue = {4},
   journal = {Journal of the Institution of Civil Engineers},
   month = {2},
   pages = {133-156},
   title = {Turbulent flow in pipes, with particular reference to the transition region between the smooth pipe and rough pipe laws},
   volume = {11},
   year = {1939},
}

@article{Ejo2020,
   abstract = {This work is on the optimal design of gas flowline system. The study involves designing of an optimal configuration of a pipeline system with optimum pipeline size and optimum number of compressors. The pipeline system used as a case study consists of three pipelines (pipeline A, 1,6km, pipeline B, 2 km, and pipeline C, 2.4 km) to a field manifold and eventually to a central processing facility (CPF) via a bulk line which is 75 km to the CPF. The optimization techniques employed in this study were iterative estimation method and Artificial Bee Colony (ABC) algorithm method. Upheaval buckling analysis, on-bottom stability analysis and pipeline end expansion analysis were further performed on the flowline system to ensure that it is strong enough to contain and transport the Non-Associated Gas (NAG) the CPF while satisfying the life span requirement and at minimum cost of investment and operation. The results shows that the optimal pipe diameter for the three hookup pipelines and the bulkline considered are 6", 6", 8", and 20" respectively. The upheaval buckling analysis results show that the flowlines are not at risk from upheaval buckling at a burial depth of 1.2m with safety factors greater than1.5 for all imperfection heights. The on-bottom stability analysis results show that the flowlines are stable and the wall thicknesses are sufficient for the attainment of a negative buoyancy effect at the swamp sections and shall not require concrete coating. The optimal number of compressor stations analysis shows that 1 intermediate compressor is needed to effectively move the gas via the bulkline from the field manifold to the CPF. The analysis could be applied to other pipeline systems.},
   author = {E K Ejomarie and E G Saturday},
   issn = {ISSN 2320-9186},
   issue = {5},
   journal = {Global Scientific Journals},
   keywords = {Artificial bee colony,End expansion analysis,On-bottom stability,Optimal number of compressors,Upheaval buckling},
   pages = {918-933},
   title = {Optimal Design of Gas Pipeline Transmission Network},
   volume = {8},
   url = {www.globalscientificjournal.com},
   year = {2020},
}
@article{Christidis2021,
   abstract = {Domestic energy consumption in the United Kingdom depends on both meteorological and socio-economic factors. The former are dominated by the effect of temperature during the colder months of the year, with the energy demand increasing as the temperature decreases. Warming of the UK climate under the influence of anthropogenic forcings is therefore expected to lead to a reduction in domestic energy consumption. Here, we present an end-to-end attribution study that investigates whether the anthropogenic effect on consumption is already evident in the United Kingdom. We analyse data of gas and electricity use in UK households during 2008–2019 and use a simple linear model to express the temperature dependence. Uncertainties in the resulting transfer functions are derived with a recent methodology, originally introduced for downscaling purposes, but adapted here for use in impact studies. The transfer functions are applied to temperature data from simulations with and without the effect of human influence on the climate, generated by 11 state-of-the-art climate models. We thus assess the anthropogenic impact on energy consumption during the reference period by comparing it with what it might have been in a climate without anthropogenic climate change, but at the same level of adaptation. We find that without human influence on the climate, UK households would consume on average about 1,400 kWh more per year, which would increase the annual energy bills by about 70 GBP. Our attribution assessment provides useful evidence of an impact that has already emerged, which can help inform UK's adaptation plans as the climate continues to warm.},
   author = {Nikolaos Christidis and Mark McCarthy and Peter A. Stott},
   doi = {10.1002/asl.1062},
   issn = {1530261X},
   issue = {11},
   journal = {Atmospheric Science Letters},
   keywords = {UK energy consumption,climate change attribution,climate change impacts,general circulation models},
   month = {11},
   publisher = {John Wiley and Sons Ltd},
   title = {Recent decreases in domestic energy consumption in the United Kingdom attributed to human influence on the climate},
   volume = {22},
   year = {2021},
}
@article{saty2018,
   abstract = {The operating principle of condensing boilers is based on exploiting heat from flue gases to pre-heat cold water at the inlet of the boiler: by condensing into liquid form, flue gases recover their latent heat of vaporization, leading to 10–12% increased efficiency with respect to traditional boilers. However, monitoring the energy efficiency of condensing boilers is complex due to their nonlinear dynamics: currently, (static) nonlinear efficiency curves of condensing boilers are calculated at quasi-stationary regime and ‘a posteriori’, i.e. from data collected during chamber tests: therefore, with this static approach, it is possible to monitor the energy efficiency only at steady-state regime. In this work we propose a novel model-based monitoring approach for condensing boilers that extends the operating regime for which monitoring is possible: the approach is based on a hybrid dynamic model of the condensing boiler, where state-dependent switching accounts for dynamically changing condensing/non condensing proportions. Monitoring the energy efficiency over the boiler's complete dynamic regime is possible via switching estimators designed for the different condensing/non condensing modes. By using real-world boiler efficiency data we show that the proposed approach results in a (dynamic) nonlinear efficiency curve which gives a more complete description of the condensing boilers operation than static nonlinear efficiency curves: in addition, the dynamic curve can be derived ‘a priori’, i.e. from first principles, or from data collected during normal boiler operation (without requiring special chamber tests).},
   author = {Harish Satyavada and Simone Baldi},
   doi = {10.1016/j.energy.2017.09.124},
   issn = {03605442},
   journal = {Energy},
   keywords = {Condensing boiler,Dynamic monitoring,Hybrid modelling,Multiple-model estimation,State-dependent switching},
   month = {1},
   pages = {121-129},
   publisher = {Elsevier Ltd},
   title = {Monitoring energy efficiency of condensing boilers via hybrid first-principle modelling and estimation},
   volume = {142},
   year = {2018},
}
@article{zeyu2020,
   abstract = {The discharge in a full flow regime represents the discharge capacity of a vertical pipe, and the Darcy. Weisbach friction factor (λ) is an important variable to calculate discharge. Since all existing equations for λ contain the Reynolds number (Re), it is problematic if the velocity is unknown. In this study, the performance of existing equations collected from studies on vertical pipes is assessed, and an approximation for the λ of vertical pipes in the full flow regime, without Re, is proposed. The performance of the Brkić and Praks equation is the best, with a maximum relative error (MRE) of 0.003% (extremely accurate). The MRE of the new approximation is 0.43%, and its assessment level is very accurate. This work is expected to provide a reference for the design and investigation of the drainage of vertical pipes.},
   author = {Zhang Zeyu and Chai Junrui and Li Zhanbin and Xu Zengguang and Li Peng},
   doi = {10.2166/ws.2020.048},
   issn = {16070798},
   issue = {4},
   journal = {Water Science and Technology: Water Supply},
   keywords = {Colebrook equation,Friction factor,Rough pipe,Smooth pipe,Vertical pipe},
   month = {6},
   pages = {1321-1333},
   publisher = {IWA Publishing},
   title = {Approximations of the darcy-weisbach friction factor in a vertical pipe with full flow regime},
   volume = {20},
   year = {2020},
}
@article{Tabkhi2008,
   author = {Firooz Tabkhi and Catherine Azzaro‐Pantel and Luc Pibouleau and Serge Domenech},
   journal = {International Journal of Hydrogen Energy},
   pages = {6222-6231},
   title = {A mathematical framework for modelling and evaluating natural gas pipeline networks under hydrogen injection},
   volume = {33},
   url = {https://api.semanticscholar.org/CorpusID:53503187},
   year = {2008},
}
@misc{King2005,
   author = {Nick King},
   title = {Levels of Carbon Dioxide (CO 2 ) and Nitrogen (N 2 ) in the NTS},
   url = {https://www.nationalgas.com/document/144896/download},
   year = {2005},
}
@misc{natgas2023,
   author = {NationalGas},
   title = {Gas Ten Year Statement Network capability >},
   url = {https://www.nationalgas.com/insight-and-innovation/gas-ten-year-statement-gtys},
   year = {2023},
}
@misc{Liebreich2021,
   author = {Michael Liebreich},
   institution = {Liebreich associates},
   month = {8},
   title = {The Clean Hydrogen Ladder [Now updated to V4.1] - liebreich},
   url = {https://www.liebreich.com/the-clean-hydrogen-ladder-now-updated-to-v4-1/},
   year = {2021},
}
@article{Bennet2017,
   abstract = {Understanding shear stress at the heat transfer surface is important because sufficient shear stress (velocity) is known to mitigate heat transfer fouling. This comprehensive paper references much literature and complements it by deriving, generally starting from force balances, the equations for calculating isothermal shear stress at the wall for single-phase flow in all of the common heat exchanger geometries: tubular, annular, parallel plate (with and without corrugations), shell-side longitudinal/window, and shell-side cross. The method for shell-side flow crossing a bundle was based upon computational fluid dynamics simulations that computed the skin friction drag on the heat transfer surface as a fraction of the total pressure drop in a cross-pass.},
   author = {Christopher A Bennett and Robert P Hohmann},
   doi = {10.1080/01457632.2016.1211913},
   issn = {0145-7632},
   issue = {9},
   journal = {Heat Transfer Engineering},
   month = {6},
   pages = {829-840},
   publisher = {Taylor & Francis},
   title = {Methods for Calculating Shear Stress at the Wall for Single-Phase Flow in Tubular, Annular, Plate, and Shell-Side Heat Exchanger Geometries},
   volume = {38},
   url = {https://doi.org/10.1080/01457632.2016.1211913},
   year = {2017},
}
@misc{NIST2001,
   author = {NIST},
   journal = {NIST Chemistry WebBook, SRD 69},
   title = {REFPROP - Hydrogen},
   url = {https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Units=SI&Type=JANAFG&Table=on#JANAFG},
   year = {2001},
}

@misc{Huber2022,
   abstract = {The NIST REFPROP software program is a powerful tool for calculating thermophysical properties of industrially important fluids, and this manuscript describes the models implemented in, and features of, this software. REFPROP implements the most accurate models available for selected pure fluids and their mixtures that are valid over the entire fluid range including gas, liquid, and supercritical states, with the goal of uncertainties approaching the level of the underlying experimental data. The equations of state for thermodynamic properties are primarily of the Helmholtz energy form; a variety of models are implemented for the transport properties. We document the models for the 147 fluids included in the current version. A graphical user interface generates tables and provides extensive plotting capabilities. Properties can also be accessed through third-party apps or user-written code via the core property subroutines compiled into a shared library. REFPROP disseminates international standards in both the natural gas and refrigeration industries, as well as standards for water/steam.},
   author = {Marcia L. Huber and Eric W. Lemmon and Ian H. Bell and Mark O. McLinden},
   doi = {10.1021/acs.iecr.2c01427},
   issn = {15205045},
   issue = {42},
   journal = {Industrial and Engineering Chemistry Research},
   month = {10},
   pages = {15449-15472},
   publisher = {American Chemical Society},
   title = {The NIST REFPROP Database for Highly Accurate Properties of Industrially Important Fluids},
   volume = {61},
   url = {https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Units=SI&Mask=1#Thermo-Gas},
   year = {2022},
}

@article{coolprop,
    author = {Bell, Ian H. and Wronski, Jorrit and Quoilin, Sylvain and Lemort, Vincent},
    title = {Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and
             the Open-Source Thermophysical Property Library CoolProp},
    journal = {Industrial \& Engineering Chemistry Research},
    volume = {53},
    number = {6},
    pages = {2498--2508},
    year = {2014},
    doi = {10.1021/ie4033999},
    URL = {http://pubs.acs.org/doi/abs/10.1021/ie4033999},
    eprint = {http://pubs.acs.org/doi/pdf/10.1021/ie4033999}
    }
@article{Yang2009,
   author = {Bobby H. Yang and Daniel D. Joseph},
   doi = {10.1080/14685240902806491},
   issn = {1468-5248},
   journal = {Journal of Turbulence},
   month = {1},
   pages = {N11},
   title = {Virtual Nikuradse},
   volume = {10},
   year = {2009},
}
@misc{toolbox,
   author = {EngineeringToolBox},
   title = {Engineering Toolbox- Thermophysical Properties},
   url = {https://www.engineeringtoolbox.com/butane-d_1415.html},
   year = {2001},
}
@article{Hassan20,
   abstract = { The use of hydrogen (H 2 ) as a substitute for fossil fuel, which accounts for the majority of the world’s energy, is environmentally the most benign option for the reduction of CO 2 emissions. This will require gigawatt-scale storage systems and as such, H 2 storage in porous rocks in the subsurface will be required. Accurate estimation of the thermodynamic and transport properties of H 2 mixed with other gases found within the storage system is therefore essential for the efficient design for the processes involved in this system chain. In this study, we used the established and regarded GERG-2008 Equation of State (EoS) and SuperTRAPP model to predict the thermo-physical properties of H 2 mixed with CH 4 , N 2 , CO 2 , and a typical natural gas from the North-Sea. The data covers a wide range of mole fraction of H 2 (10–90 Mole%), pressures (0.01–100 MPa), and temperatures (200–500 K) with high accuracy and precision. Moreover, to increase ease of access to the data, a user-friendly software (H2Themobank) is developed and made publicly available. },
   author = {Aliakbar Hassanpouryouzband and Edris Joonaki and Katriona Edlmann and Niklas Heinemann and Jinhai Yang},
   doi = {10.1038/s41597-020-0568-6},
   issn = {2052-4463},
   issue = {1},
   journal = {Scientific Data},
   month = {7},
   pages = {222},
   title = {Thermodynamic and transport properties of hydrogen containing streams},
   volume = {7},
   year = {2020},
}
@misc{H2Blends21,
   author = {NationalGrid},
   title = {Hydrogen Blends in the NTS -  A theoretical exploration Gas Future Operability Planning},
   url = {https://www.nationalgas.com/document/137506/download},
   year = {2021},
}
@article{Pomerenk2023,
   author = {Olivia Pomerenk and Simon Carrillo Segura and Fangning Cao and Jiajie Wu and Leif Ristroph},
   journal = {Journal of Fluid Mechanics},
   title = {Hydrodynamics of finite-length pipes at intermediate Reynolds numbers},
   volume = {959},
   url = {https://api.semanticscholar.org/CorpusID:257666311},
   year = {2023},
}
@misc{DESNZ2023,
   author = {DESNZ},
   journal = {https://www.gov.uk/government/statistics/dukes-calorific-values},
   month = {7},
   title = {Digest of UK Energy Statistics (DUKES): calorific values and density of fuels},
   year = {2023},
}
@misc{ehs21,
   author = {DLUHC},
   journal = {https://www.gov.uk/government/statistics/english-housing-survey-2021-to-2022-headline-report/english-housing-survey-2021-to-2022-headline-report},
   title = {English Housing Survey 2021 to 2022: headline report},
   url = {https://www.gov.uk/government/statistics/english-housing-survey-2021-to-2022-headline-report/english-housing-survey-2021-to-2022-headline-report},
   year = {2022},
}
@article{Terry2023,
abstract = {Heating with air source heat pumps is an important step in the transition to Net Zero Carbon for many households. When predicting the resulting energy consumption it is frequently assumed that heating demand is unchanged from before the switch from gas heating. However, heating patterns usually change, with a net increase in heating demand to achieve the same level of thermal comfort. This paper discusses the change in pattern and uses modelling to determine the resulting increase in energy demand. It shows that the increase can be predicted with information about dwelling heat loss and thermal mass as well as weather and the pre and post switch heating pattern. The increase can be 20% or more for homes with high heat loss. It proposes that the metric for heating demand increase would be a useful measure of heat pump readiness, and that the parameters required to assess this should be provided on energy performance certificates.},
author = {Terry, Nicola and Galvin, Ray},
doi = {10.1016/j.enbuild.2023.113183},
file = {:C\:/Users/philip/Downloads/terry-S0378778823004139.htm:htm},
issn = {03787788},
journal = {Energy and Buildings},
month = {aug},
pages = {113183},
title = {{How do heat demand and energy consumption change when households transition from gas boilers to heat pumps in the UK}},
url = {https://linkinghub.elsevier.com/retrieve/pii/S0378778823004139},
volume = {292},
year = {2023}
}
@misc{ARUP2023,
   author = {ARUP},
   title = {Future of Great Britain's Gas Networks - Final Report for National Infrastructure Commission and Ofgem},
   url = {https://nic.org.uk/app/uploads/Arup-Future-of-UK-Gas-Networks-18-October-2023.pdf},
   year = {2023},
}
@article{Duchowny22,
   abstract = {Natural gas is an essential energy source for a large variety of applications in today's society. Forecasts predict that it will also play a vital role in decarbonizing the energy sector. The price of natural gas is determined by its calorific value, which depends on its composition. Thus, the accurate composition quantification of natural gas is essential to both producers and consumers. In this context, proton Nuclear Magnetic Resonance (NMR) spectra of pure and mixed hydrocarbon gases have been studied with a desktop spectrometer in the range between 1 to 200 bar. Although spectral linewidth, chemical shift, and relaxation times depend on pressure and composition, the hydrocarbon concentrations of binary, ternary, and an eleven-component gas mixture could be quantified by spectral analysis using Spectral Hard Modeling, a mechanistical multivariate regression. Concentrations determined for the main components at 200 bar by NMR deviated by 0.1 to 0.18 mol% from Gas Chromatography values, leading to a difference in calorific values of only 0.15%. The presented results demonstrate that benchtop NMR combined with chemometrics can quantify complex hydrocarbon-gas mixtures reliably and quickly.},
   author = {Anton Duchowny and Oliver Mohnke and Holger Thern and Pablo Matias Dupuy and Hege Christin Widerøe and Audun Faanes and Anfinn Paulsen and Markus Küppers and Bernhard Blümich and Alina Adams},
   doi = {10.1016/j.egyr.2022.02.289},
   issn = {23524847},
   journal = {Energy Reports},
   keywords = {Energy management,High-pressure,Indirect Hard Modeling,Natural gas quantification,Nuclear Magnetic Resonance},
   month = {11},
   pages = {3661-3670},
   publisher = {Elsevier Ltd},
   title = {Composition analysis of natural gas by combined benchtop NMR spectroscopy and mechanistical multivariate regression},
   volume = {8},
   year = {2022},
}
@misc{Altfeld2013,
   abstract = {There are proposals to inject hydrogen (H2) from renewable sources in the natural gas network. This measure would allow the very large transport and storage capacities of the existing infrastructure, particularly high-pressure pipelines, to be used for indirect electricity transport and storage. 1. SUMMARY AND CONCLUSIONS The results of this study show that an admixture of up to 10 % by volume of hydrogen to natural gas is possible in some parts of the natural gas system. However there are still some important areas where issues remain: ■ underground porous rock storage: hydrogen is a good substrate for sulphate-reducing and sulphur-reducing bacteria. As a result, there are risks associated with: bacterial growth in underground gas storage facilities leading to the formation of H2S; the consumption of H2, and the plugging of reservoir rock. A limit value for the maximum acceptable hydrogen concentration in natural gas cannot be defined at the moment. (H2-related aspects concerning wells have not been part of this project); ■ steel tanks in natural gas vehicles: specification UN ECE R 110 stipulates a limit value for hydrogen of 2 vol%; ■ gas turbines: most of the currently installed gas turbines were specified for a H2 fraction in natural gas of 1 vol% or even lower. 5 % may be attainable with minor modification or tuning measures. Some new or upgraded types will be able to cope with concentrations up to 15 vol%; ■ gas engines: it is recommended to restrict the hydrogen concentration to 2 vol%. Higher concentrations up to 10 vol% may be possible for dedicated gas engines with sophisticated control systems if the methane number of the natural gas/hydrogen mixture is well above the specified minimum value; ■ many process gas chromatographs will not be capable of analysing hydrogen. Investigations have been conducted to evaluate the impact of hydrogen as related to the above topics. At present it is not possible to specify a limiting hydrogen value which would generally be valid for all parts of the European gas infrastructure and, as a consequence, we strongly recommend a case by case analysis. Some practical recommendations are given at the end of the paper.},
   author = {Klaus Altfeld and Dave Pinchbeck},
   title = {Admissible hydrogen concentrations in natural gas systems},
   url = {www.gas-for-energy.com https://www.gerg.eu/wp-content/uploads/2019/10/HIPS_Final-Report.pdf},
   year = {2013},
}
@misc{ISO6976,
   author = {ISO},
   title = {ISO 6976:2016 Natural gas Calculation of calorific values, density, relative density and Wobbe indices from composition},
   url = {https://www.iso.org/standard/55842.html},
   year = {2022},
}
@misc{ISO13443,
   author = {ISO},
   journal = {https://www.iso.org/standard/20461.html},
   title = {ISO 13443:1996 Natural gas
Standard reference conditions},
   url = {https://www.iso.org/standard/20461.html},
   year = {2020},
}
@misc{GS(M)2023,
   author = {Statute, 284},
   abstract = {In this Schedule, the reference conditions are 15C and 1.01325 bar},
   journal = {https://www.legislation.gov.uk/uksi/2023/284/made},
   month = {3},
   title = {The Gas Safety (Management) (Amendment) Regulations 2023},
   year = {2023},
}
@misc{Laughton2019,
   author = {Andrew Laughton},
   title = {Calculation of Calorific Values and Relative Density from Gas Composition (Update of ISO 6976)},
   url = {https://www.researchgate.net/publication/332963484},
   year = {2019},
}
@unpublished{Lander2017,
   author = {David Lander and Tony Humphreys},
   note = "From UK gas governance website.",
   title = {The revision of ISO-6976 and assessment of the impacts of changes},
   url = {https://nfogm.no/wp-content/uploads/2019/02/2017-26-The-revision-of-ISO-6976-and-assessment-of-the-impacts-of-changes-Lander-Dave-Lander-Consulting.pdf},
   year = {2017},
}
@misc{utonomy23,
   author = {UTONOMY},
   journal = {https://utonomy.co.uk/pressure-management/},
   title = {UTONOMY’S PRESSURE MANAGEMENT},
   url = {https://utonomy.co.uk/pressure-management/},
   year = {2023},
}
@misc{cngservices2019,
   author = {cngservices},
   title = {Fordoun Case Study},
   url = {https://www.cngservices.co.uk/case_study/fordoun/},
   year = {2019},
}
@misc{WikiVP2024,
abstract = {The vapor pressure of water is the pressure exerted by molecules of water vapor in gaseous form (whether pure or in a mixture with other gases such as air). The saturation vapor pressure is the pressure at which water vapor is in thermodynamic equilibrium with its condensed state. At pressures higher than vapor pressure, water would condense, while at lower pressures it would evaporate or sublimate. The saturation vapor pressure of water increases with increasing temperature. The boiling point of water is the temperature at which the saturated vapor pressure equals the ambient pressure.},
author = {Wikipedia},
title = {{Vapour pressure of water}},
url = {https://en.wikipedia.org/wiki/Vapour_pressure_of_water},
urldate = {2024-01-03},
year = {2024}
}
@article{Picard_2008,
doi = {10.1088/0026-1394/45/2/004},
url = {https://dx.doi.org/10.1088/0026-1394/45/2/004},
year = {2008},
month = {feb},
publisher = {},
volume = {45},
number = {2},
pages = {149},
author = {A Picard and R S Davis and M Gläser and K Fujii},
title = {Revised formula for the density of moist air (CIPM-2007)},
journal = {Metrologia},
abstract = {Measurements of air density determined gravimetrically and by using the CIPM-81/91 formula, an equation of state, have a relative deviation of 6.4 × 10−5.

This difference is consistent with a new determination of the mole fraction of argon xAr carried out in 2002 by the Korea Research Institute of Standards and Science (KRISS) and with recently published results from the LNE. The CIPM equation is based on the molar mass of dry air, which is dependent on the contents of the atmospheric gases, including the concentration of argon. We accept the new argon value as definitive and amend the CIPM-81/91 formula accordingly. The KRISS results also provide a test of certain assumptions concerning the mole fractions of oxygen and carbon dioxide in air. An updated value of the molar gas constant R is available and has been incorporated in the CIPM-2007 equation. In making these changes, we have also calculated the uncertainty of the CIPM-2007 equation itself in conformance with the Guide to the Expression of Uncertainty in Measurement, which was not the case for previous versions of this equation. The 96th CIPM meeting has accepted these changes.}
}
@article{Zhao2023,
abstract = {The presence of non-condensable gas significantly deteriorates the heat transfer of water vapor condensation, and the condensation process of a CO2-water vapor mixture is a key process for CO2 separation and purification. In this study, we fabricated an experimental system to investigate the heat transfer characteristics of the mixed vapor condensation using a CO2 mass fraction of 32–85 % at a flow rate of 0.4–1.2 m/s and surface subcooling of 3–70 K. Liquid film thickness, gas-phase diffusion layer thickness, and interface temperature were calculated based on the double boundary layer model. An increase in the flow rate (0.4–1.2 m/s) improved the mixed vapor heat transfer, especially at high CO2 concentrations, and increased the heat transfer coefficient by 177 % at a surface subcooling of 40 K. The lower the surface subcooling, the greater the effect of the flow rate. As the CO2 concentration increased, the liquid film thickness decreased, whereas the gas-phase diffusion layer thickness increased to 1.7 mm, which is tens and hundreds of times the liquid film thickness. Increasing the flow rate slightly changed the liquid film but significantly reduced the gas-phase diffusion layer thickness and afforded a better heat transfer performance. A new heat transfer correlation equation was fitted, and the error of the predicted value was within ± 30 %. This study provides fundamental insights for the design of related heat exchangers.},
author = {Zhao, Yulong and Diao, Hongmei and Qin, Yao and Xie, Liyao and Ge, Minghui and Wang, Yulin and Wang, Shixue},
doi = {https://doi.org/10.1016/j.applthermaleng.2023.120557},
issn = {1359-4311},
journal = {Applied Thermal Engineering},
keywords = {CO-water vapor mixture,Condensation heat transfer,Flow rate,Gas- phase diffusion layer,Liquid film},
pages = {120557},
title = {{Effect of flow rate on condensation of CO2-water vapor mixture on a vertical flat plate}},
url = {https://www.sciencedirect.com/science/article/pii/S1359431123005860},
volume = {229},
year = {2023}
}
@article{Diamond2003,
title = {Solubility of CO2 in water from −1.5 to 100 °C and from 0.1 to 100 MPa: evaluation of literature data and thermodynamic modelling},
journal = {Fluid Phase Equilibria},
volume = {208},
number = {1},
pages = {265-290},
year = {2003},
issn = {0378-3812},
doi = {https://doi.org/10.1016/S0378-3812(03)00041-4},
url = {https://www.sciencedirect.com/science/article/pii/S0378381203000414},
author = {Larryn W. Diamond and Nikolay N. Akinfiev},
keywords = {Carbon dioxide, HO, Model, Equation of state, Vapour–liquid equilibria, Activity coefficient},
abstract = {Experimental measurements of the solubility of CO2 in pure water at pressures above 1MPa have been assembled from 25 literature studies and tested for their accuracy against simple thermodynamic criteria. Of the 520 data compiled, 158 data were discarded. Possible reasons for the observed discrepancies between datasets are discussed. The 362 measurements that satisfy the acceptance criteria have been correlated by a thermodynamic model based on Henry’s law and on recent high-accuracy equations of state. The assumption that the activity coefficients of aqueous CO2 are equal to unity is found to be valid up to solubilities of approximately 2mol%. At higher solubilities the activity coefficients show a systematic trend from values greater than unity at low temperatures, to values progressively lower than unity at high temperatures. An empirical correction function that describes this trend is applied to the basic model. The resulting corrected model reproduces the accepted experimental solubilities with a precision of better than 2% (1 standard deviation) over the entire P–T–x range considered, whereas the data themselves scatter with a standard deviation of approximately 1.7%. The model is available as a computer code at <www.geo.unibe.ch/diamond>. In addition to calculating solubility, the code calculates the full set of partial molar properties of the CO2-bearing aqueous phase, including activity coefficients, partial molar volumes and chemical potentials.}
}
@article{Carroll1991,
   abstract = {The system carbon dioxide‐water is of great scientific and technological importance. Thus, it has been studied often. The literature for the solubility of carbon dioxide in water is vast and interdisciplinary. An exhaustive survey was conducted and approximately 100 experimental investigations were found that reported equilibrium data at pressures below 1 MPa. A model based on Henry’s law was used to correlate the low pressure data (those up to 1 MPa). The following correlation of the Henry’s constants (expressed on a mole fraction basis) was developed ln(H21/MPa)=−6.8346+1.2817×104/T−3.7668×106/T2 +2.997×108/T3 The correlation is valid for 273<T<433 K(0<t<160 °C) where T is in K. Any experimental data that deviated significantly from this model were duly noted.},
   author = {John J Carroll and John D Slupsky and Alan E Mather},
   doi = {10.1063/1.555900},
   issn = {0047-2689},
   issue = {6},
   journal = {Journal of Physical and Chemical Reference Data},
   month = {11},
   pages = {1201-1209},
   title = {The Solubility of Carbon Dioxide in Water at Low Pressure},
   volume = {20},
   url = {https://doi.org/10.1063/1.555900},
   year = {1991},
}
@book{MacKay2008,
  title={Sustainable Energy - Without the Hot Air},
  author={David J. C. MacKay},
  year={2008},
    publisher = {UIT Cambridge Ltd.},
url={https://www.withouthotair.com/cE/page_303.shtml}
}1.561}, # 
    'C3H8': {'Tc': 369.15, 'Pc': 42.48, 'omega': 0.1521, 'Mw': 44.096, 'Vs': (8.2,300, 0.93), 'Hc':2.22}, # https://www.engineeringtoolbox.com/propane-d_1423.html
    'nC4': {'Tc': 425, 'Pc': 38,  'omega': 0.20081, 'Mw': 58.1222, 'Vs': (7.5,300, 0.950), 'Hc':2.8781}, # omega http://www.coolprop.org/fluid_properties/fluids/n-Butane.html https://www.engineeringtoolbox.com/butane-d_1415.html 
    'iC4': {'Tc': 407.7, 'Pc': 36.5, 'omega': 0.1835318, 'Mw': 58.1222, 'Vs': (7.5,300, 0.942), 'Hc':2.86959}, # omega  http://www.coolprop.org/fluid_properties/fluids/IsoButane.html https://webbook.nist.gov/cgi/cbook.cgi?ID=C75285&Mask=1F https://webbook.nist.gov/cgi/cbook.cgi?Name=butane&Units=SI Viscocity assumed same as nC4
    'nC5': {'Tc': 469.8, 'Pc': 33.6, 'omega': 0.251032, 'Mw': 72.1488, 'Vs': (6.7,300, 1.0), 'Hc':3.509}, # omega http://www.coolprop.org/fluid_properties/fluids/n-Pentane.html     
    'iC5': {'Tc': 461.0, 'Pc': 33.8, 'omega': 0.2274, 'Mw': 72.1488, 'Vs': (6.7,300, 0.94), 'Hc':3.509}, # omega http://www.coolprop.org/fluid_properties/fluids/Isopentane.html  Viscocity assumed same as nC5 
    
    'neoC5': {'Tc': 433.8, 'Pc': 31.963, 'omega': 0.1961, 'Mw': 72.1488, 'Vs': (6.9326,300, 0.937), 'Hc':3.509},
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C463821&Units=SI&Mask=4#Thermo-Phase
    # omega from http://www.coolprop.org/fluid_properties/fluids/Neopentane.html
    
    'C6':  {'Tc': 507.6, 'Pc': 30.2, 'omega': 0.1521, 'Mw': 86.1754, 'Vs': (8.6,400, 1.03), 'Hc':4.163}, # omega is 0.2797 isohexane    
    'CO2': {'Tc': 304.2, 'Pc': 73.8, 'omega': 0.228, 'Mw': 44.01, 'Vs': (15.0,300, 0.872), 'Hc':0}, # https://en.wikipedia.org/wiki/Acentric_factor
    'H2O': {'Tc': 647.1, 'Pc': 220.6, 'omega': 0.344292, "Mw": 18.015, 'Vs': (9.8,300, 1.081), 'Hc':0}, # https://link.springer.com/article/10.1007/s10765-020-02643-6/tables/1
    'N2': {'Tc': 126.21, 'Pc': 33.958, 'omega': 0.0372, 'Mw':28.013, 'Vs': (17.9,300, 0.658), 'Hc':0}, #  omega http://www.coolprop.org/fluid_properties/fluids/Nitrogen.html
    'He': {'Tc': 5.2, 'Pc': 2.274, 'omega': -0.3836, 'Mw': 4.0026, 'Vs': (19.9,300, 0.69), 'Hc':0},  # omega http://www.coolprop.org/fluid_properties/fluids/Helium.html
    # https://eng.libretexts.org/Bookshelves/Chemical_Engineering/Distillation_Science_(Coleman)/03%3A_Critical_Properties_and_Acentric_Factor
    # N2 https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.2c00363/suppl_file/ie2c00363_si_001.pdf
    # N2 omega is from https://en.wikipedia.org/wiki/Acentric_factor
    'Ar': {'Tc': 150.687, 'Pc': 48.630, 'omega': 0, 'Mw': 39.948, 'Vs': (22.7,300, 0.77), 'Hc':0}, #https://en.wikipedia.org/wiki/Acentric_factor
    'O2': {'Tc': 154.581, 'Pc': 50.43, 'omega': 0.022, 'Mw': 31.9988, 'Vs': (20.7,300, 0.72), 'Hc':0},# http://www.coolprop.org/fluid_properties/fluids/Oxygen.html
    }

# Natural gas compositions (mole fractions)
gas_mixtures = {
    'Groening': {'CH4': 0.813, 'C2H6': 0.0285, 'C3H8': 0.0037, 'nC4': 0.0014, 'nC5': 0.0004, 'C6': 0.0006, 'CO2': 0.0089, 'N2': 0.1435, 'O2': 0}, # Groeningen gas https://en.wikipedia.org/wiki/Groningen_gas_field
    
    'Biomethane': {'CH4': 0.92,  'C3H8': 0.04, 'CO2': 0.04 }, # wobbe central, not a real natural gas  https://www.gasgovernance.co.uk/sites/default/files/ggf/Impact%20of%20Natural%20Gas%20Composition%20-%20Paper_0.pdf
 
    '10C2-10N': {'CH4': 0.80,  'C3H8': 0.1, 'N2': 0.1 }, # RH corner of allowable wobbe polygon ?
    '7C2-2N': {'CH4': 0.91,  'C3H8': 0.07, 'N2': 0.02 }, # top corner of allowable wobbe polygon ?
  
    'mix6': {'CH4': 0.8, 'C2H6': 0.05, 'C3H8': 0.03, 'CO2': 0.02, 'N2': 0.10}, # ==mix6 from      https://backend.orbit.dtu.dk/ws/files/131796794/FPE_D_16_00902R1.pdf - no, somewhere else..

    'NTS79': {'CH4': 0.9363, 'C2H6': 0.0325, 'C3H8': 0.0069, 'nC4': 0.0027, 'CO2': 0.0013, 'N2': 0.0178, 'He': 0.0005, 'nC5': 0.002}, # https://en.wikipedia.org/wiki/National_Transmission_System
    # https://en.wikipedia.org/wiki/National_Transmission_System
    # This NTS composition from Wikipedia actually comes from 1979 !  Cassidy, Richard (1979). Gas: Natural Energy. London: Frederick Muller Limited. p. 14.
    
   'Fordoun': {'CH4': 0.895514, 'C2H6': 0.051196, 'C3H8': 0.013549, 'iC4': 0.001269, 'nC4': 0.002162, 'neoC5': 2e-05, 'iC5': 0.000344, 'nC5': 0.003472, 'C6': 0.002377, 'CO2': 0.020743, 'N2': 0.009354}, # Normalized.email John Baldwin 30/12/2023
    # 'Fordoun': { 'CH4':  0.900253, 'C2H6':  0.051467, 'C3H8':  0.013621, 'iC4':  0.001276, 'nC4':  0.002173, 'neoC5':  0.000020,'iC5':  0.000346, 'nC5':  0.003490,  'C6':  0.002390, 'CO2':  0.020853, 'N2':  0.009404, }, # original email John Baldwin 30/12/2023

    '11D': { 'CH4':  0.88836, 'C2H6':  0.04056, 'C3H8':  0.00997, 'iC4':  0.00202, 'nC4':  0.00202, 'iC5':  0.00050, 'nC5':  0.00050, 'neoC5':  0.00050, 'C6':  0.00049, 'CO2':  0.01512, 'N2':  0.03996, }, # normlized 11D gas from Duchowny22, doi:10.1016/j.egyr.2022.02.289
    
    'Algerian': {'CH4': 0.867977, 'C2H6': 0.085862, 'C3H8': 0.011514, 'iC4': 0.000829, 'nC4': 0.001044, 'iC5': 0.000205, 'nC5': 0.000143, 'C6': 0.000164, 'CO2': 0.018505, 'N2': 0.012927, 'He': 0.000829}, # NORMALIZED # Algerian NG, Romeo 2022, C6+
 
    'North Sea': {'CH4': 0.836, 'C2H6': 0.0748, 'C3H8':0.0392, 'nC4':0.0081, 'iC4':0.0081, 'nC5':0.0015, 'iC5':0.0014, 'CO2':0.0114, 'N2':0.0195}, # North Sea gas [Hassanpou] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7347886/
        
    # 'ethane': {'C2H6': 1.0}, # ethane, but using the mixing rules: software test check
    # 'propane': {'C3H8': 1.0}, # ethane, but using the mixing rules: software test check
    
    # dry air from Picard2008, who is quoting 
    'dryAir':  {'N2': 0.780848, 'O2': 0.209390, 'CO2': 0.00040, 'Ar': 0.009332},
    #'Air':  {'N2': 0.761749, 'O2': 0.204355, 'CO2': 0.00039, 'Ar': 0.009112, 'He': 5.0e-06, 'H2O': 0.024389} # https://www.thoughtco.com/chemical-composition-of-air-604288
    # But ALSO adding 2.5% moisture to the air and normalising
}

"""reduce the lower limit for Wobbe Index from 47.2 MJ/m³  to 46.50 MJ/m³ was approved by HSE. 
This shall enter into force from 6 April 2025
"Gas Ten Year Statement December 2023"
https://www.nationalgas.com/document/144896/download
"""
gas_mixture_properties = {
    'Algerian': {'Wb': 49.992, 'HHV': 39.841, 'RD': 0.6351}, #Algerian NG, Romeo 2022, C6+ BUT not according to my calcs. for Hc and wobbe.
    'Air': {'Hc': 0} 
}


# 20% H2, remainder N.Sea gas. BUT may need adjusting to maintain Wobbe value, by adding N2 probably.
fifth = {}
fifth['H2'] = 0.2
ng = gas_mixtures['Fordoun']
for g in ng:
    fifth[g] = ng[g]*0.8
gas_mixtures['Fordoun+20%H2'] = fifth

air = {}
air['H2O'] = 0.0084 # 50% RH at 15
ag = gas_mixtures['dryAir']
for g in ag:
    air[g] = ag[g]*(1 - air['H2O'])
gas_mixtures['Air'] = air

#print(f"NatGas gas 11D composition: Duchowny22, doi:10.1016/j.egyr.2022.02.289")
print(f"NatGas at Fordoun NTS 20th Jan.2021")
nts = gas_mixtures["Fordoun"]
for f in nts:
    print(f"{f:5}\t{nts[f]*100:7.5f} %")
    
# Binary interaction parameters for hydrocarbons for Peng-Robinson
# based on the Chueh-Prausnitz correlation
# from https://wiki.whitson.com/eos/cubic_eos/
# also from Privat & Jaubert, 2023 (quoting a 1987 paper).
# Note that the default value (in the code) is -0.019 as this represents ideal gas behaviour.

# These are used in function estimate_k_?(g1, g2) which estimates these parameters from gas data.
# NOT NOW USED, instead weuse the Courtinho estimation procedure in estimate_k()
# The difference is undetectable in our use at ambient conditions.
k_ij = {
    'CH4': {'C2H6': 0.0021, 'C3H8': 0.007, 'iC4': 0.013, 'nC4': 0.012, 'iC5': 0.018, 'nC5': 0.018, 'C6': 0.021, 'CO2': 0},
    'C2H6': {'C3H8': 0.001, 'iC4': 0.005, 'nC4': 0.004, 'iC5': 0.008, 'nC5': 0.008, 'C6': 0.010},
    'C3H8': {'iC4': 0.001, 'nC4': 0.001, 'iC5': 0.003, 'nC5': 0.003, 'C6': 0.004},
    'iC4': {'nC4': 0.0, 'iC5': 0.0, 'nC5': 0.0, 'C6': 0.001}, # ?
    'nC4': {'iC5': 0.001, 'nC5': 0.001, 'C6': 0.001}, # ?
    'iC5': {'C6': -0.019}, # placeholder
    'nC5': {'C6': -0.019}, # placeholder    
    #'C6': {'C6': -0.019}, # placeholder
    'CO2': {'C6': -0.019}, # placeholder
    'H2O': {'C6': -0.019}, # placeholder
    'N2': {'C6': -0.019}, # placeholder
    'He': {'C6': -0.019}, # placeholder
    'H2': {'C6': -0.019}, # placeholder
    'O2': {'C6': -0.019}, # placeholder
    'Ar': {'C6': -0.019}, # placeholder
}

# We memoize some functions so that they do not get repeadtedly called with
# the same arguments. Yet still be retain a more obvius way of writing the program.

def memoize(func):
    """Standard memoize function to use in a decorator, see
    https://medium.com/@nkhaja/memoization-and-decorators-with-python-32f607439f84
    """
    cache = func.cache = {}
    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func

@memoize   
def estimate_k_fixed(gas1, gas2, T=298):
    """Using the data table for k_ij"""
    k = -0.019 # value for ideal solution Privat & Jaubert, 2023
    if gas2 in k_ij[gas1]:
        k = k_ij[gas1][gas2]
    if gas1 in k_ij[gas2]:
        k = k_ij[gas2][gas1]
    return k

@memoize
def estimate_k_gao(gas1, gas2, T=298):
    """An estimate of the temperature-independent binary interaction parameters, eq.(29) in 
    Privat & Jaubert 2023, which is due to Gao reworking the method of Chueh & Prausnitz (1960s)
    
    BUT temperature-independent BIPs are known to be inaccurate.
"""
    Tc1 = gas_data[gas1]["Tc"]
    Tc2 = gas_data[gas2]["Tc"]
    
    Pc1 = gas_data[gas1]["Pc"]
    Pc2 = gas_data[gas2]["Pc"]
    
    Zc1 = peng_robinson(Tc1, Pc1, gas1)
    Zc2 = peng_robinson(Tc2, Pc2, gas2)
    
    power = 0.5*(Zc1 + Zc2)
    term = 2 * np.sqrt(Tc1*Tc2) / (Tc1 + Tc2)
    k = 1 - pow(term, power)
    return k  
    
@memoize    
def estimate_k(gas1, gas2, T=298):
    """Courtinho method implemented here.
    
    BUT We should REALLY be using the temperature-dependent group-contribution method as described in
    author = {Romain Privat and Jean-Nol Jaubert},
    doi = {10.5772/35025},
    book = {Crude Oil Emulsions- Composition Stability and Characterization},
    pages = {71-106},
    publisher = {InTech},
    title = {Thermodynamic Models for the Prediction of Petroleum-Fluid Phase Behaviour},
    year = {2012},
    which has data for all the components we are dealing with (check this..)"""

    a1, b1 = a_and_b(gas1, T)
    a2, b2 = a_and_b(gas2, T)
    term = 2 * np.sqrt(b1*b2) / (b1 + b2)
    k = 1 - 0.885 * pow(term, -0.036)
    return k
    
def check_composition(mix, composition, n=0):
    """Checks that the mole fractions add up to 100%
    This gives warnings and prints out revised compositions for manual fixing,
    but after doing that, it normalises everything perfectly using float division
    so that all calculations on the data are using perfectly normalised compositions, even if they
    don't quite match what the data declaration says."""
    eps = 0.000001
    warn = 0.02 # 2 %
    
    x = 0
    norm = 1
    for gas, xi in composition.items():
       x += xi
    norm = x

    if abs(x - 1.0) > eps:
        if abs(x - 1.0) < warn:
            print(f"--------- Warning gas mixture '{mix}', {100*(1-warn)}% > {100*x:.5f} > {100*(1+warn)}%. Normalizing.")
        else:
            print(f"######### BAD gas mixture '{mix}', molar fractions add up to {x} !!!")
            
        # Normalising is not done exactly, but with rounded numbers to 6 places of decimals.
        print(f"Stated:\n   '{mix}': {gas_mixtures[mix]},") 
        for g in gas_mixtures[mix]:
            gas_mixtures[mix][g] = float(f"{gas_mixtures[mix][g]/norm:.6f}")
        print(f"Normed:\n   '{mix}': {gas_mixtures[mix]},") 
        
        # Recursive call to re-do normalization, still with 6 places of decimals.
        n += 1
        if n < 5:
            newcomp = gas_mixtures[mix]
            check_composition(mix, newcomp, n)
        else:
            print(f"Cannot normalise using rounded 6 places of decimals, doing it exactly:") 
            gas_mixtures[mix][g] = gas_mixtures[mix][g]/norm
            print(f"Normed:\n   '{mix}': {gas_mixtures[mix]},") 
        
        
    # Normalise all the mixtures perfectly, however close they already are to 100%
    for gas, xi in composition.items(): 
        x = xi/norm
        gas_mixtures[mix][gas] = x
           
def density_actual(gas, T, P):
    """Calculate density for a pure gas at temperature T and pressure = P
    """
    ϱ = P * gas_data[gas]['Mw'] / (peng_robinson(T, P, gas) * R * T)
    return ϱ

@memoize   
def viscosity_actual(gas, T, P):
    """Calculate viscosity for a pure gas at temperature T and pressure = P
    """
    if len(gas_data[gas]['Vs']) == 3:
        vs0, t, power  = gas_data[gas]['Vs'] # at T=t  
    else:
        vs0, t  = gas_data[gas]['Vs'] # at T=t 
        power = 0.5

    vs = pow(T/t, power) * vs0 # at 1 atm

    return vs

def viscosity_values(mix, T, P):

    values = {}
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # this is where we call the function to calculate the viscosity
        vs = viscosity_actual(gas, T, P) 
        values[gas] = vs
    return values

@memoize       
def do_mm_rules(mix):
    """Calculate the mean molecular mass of the gas mixture"""
    
    if mix in gas_data:
        # if a pure gas
        return gas_data[mix]['Mw']
        
    mm_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        mm_mix += x * gas_data[gas]['Mw']
    
    return mm_mix

@memoize
def linear_mix_rule(mix, values):
    """Calculate the mean value of a property for a mixture
    
    values: dict {gas1: v1, gas2: v2, gas3: v3 etc}
                 where gas1 is one of the component gases in the mix, and v1 is value for that gas
    """
    value_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        value_mix += x * values[gas]
    
    return value_mix
    
@memoize
def explog_mix_rule(mix, values):
    """Calculate the mean value of a property for a mixture
    
    values: dict {gas1: v1, gas2: v2, gas3: v3 etc}
                 where gas1 is one of the component gases in the mix, and v1 is value for that gas
                 
    This exp(log()) mixing rue was used by Xiong 2023 for the Peng-Robinson FT case. eqn.(6).
    """
    ln_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # exp(ln()) mixing rule for volume factor
        ln_mix += x * np.log(values[gas]) # natural log
    
    return np.exp(ln_mix)

@memoize
def hernzip_mix_rule(mix, values):
    """Calculate the mean value of a property for a mixture
    using the Herning & Zipper mixing rule
    
    values: dict {gas1: v1, gas2: v2, gas3: v3 etc}
                 where gas1 is one of the component gases in the mix, and v1 is value for that gas
    """
    composition = gas_mixtures[mix]
    # sum_of_sqrt(Mw)
    x = 0
    sqrt_Mw = 0
    for gas, x in composition.items():
        sqrt_Mw += x * np.sqrt(gas_data[gas]['Mw'])
 
    value_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        value_mix += x * values[gas] * np.sqrt(gas_data[gas]['Mw']) / sqrt_Mw
    
    return value_mix

        
def do_notwilke_rules(mix):
    """Calculate the mean viscosity of the gas mixture"""
    vs_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        vs, _ = gas_data[gas]['Vs'] # ignore T, so value for hexane will be bad
        vs_mix += x * vs
    
    return vs_mix

@memoize
def z_mixture_rules(mix, T):
    """
    Calculate the Peng-Robinson constants for a mixture of hydrocarbon gases.
    
    This uses the (modified) Van der Waals mixing rules and assumes that the
    binary interaction parameters are non-zero between all pairs of components
    that we have data for.
    
    Zc is the compressibility factor at the critical point    
    """
    
    # Initialize variables for mixture properties
    a_mix = 0
    b_mix = 0
    Zc_mix = 0
    
    composition = gas_mixtures[mix]
    # Calculate the critical volume and critical compressibility for the mixture
    # Vc_mix = 0
    # for gas, xi in composition.items(): 
        # Tc = gas_data[gas]['Tc']
        # Pc = gas_data[gas]['Pc']
        # Vc_mix += xi * (0.07780 * Tc / Pc)
   
    # Calculate the mixture critical temperature and pressure using mixing rules
    for gas1, x1 in composition.items():
        Tc1 = gas_data[gas1]['Tc']
        Pc1 = gas_data[gas1]['Pc']
         
        a1, b1 = a_and_b(gas1, T) 
        
        # Linear mixing rule for volume factor
        b_mix += x1 * b1
           
        # Van der Waals mixing rules for 'a' factor
        for gas2, x2 in composition.items(): # pairwise, but also with itself.
            Tc2 = gas_data[gas2]['Tc']
            Pc2 = gas_data[gas2]['Pc']
            #omega2 = gas_data[gas2]['omega']
            a2, b2 = a_and_b(gas2, T) 
            
            # Use mixing rules for critical properties
            k = estimate_k(gas1, gas2, T)
             
            a_mix += x1 * x2 * (1 - k) * (a1 * a2)**0.5  
            
       # Return the mixture's parameters for the P-R law
    return { mix: 
        {
            'a_mix': a_mix,
            'b_mix': b_mix,
         }
    }

"""This function uses simple mixing rules to calculate the mixture’s critical properties. The kij parameter, which accounts for the interaction between different gases, is assumed to be 0 for simplicity. In practice, kij may need to be adjusted based on experimental data or literature values for more accurate results.
 """


def get_LMN(omega):
    """Twu (1991) suggested a replacement for the alpha function, which instead of depending
        only on T & omega, depends on T, L, M, N (new material constants)
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8851615/pdf/ao1c06519.pdf
        
        We now have the PDF for the H2 + n-butane Jaubert et al. 2013 paper which has 
        all these parameters
    """
    # These equations are from Privat & Jaubert (2023)
    # https://www.sciencedirect.com/science/article/pii/S0378381222003168
    L = 0.0544 + 0.7536 * omega + 0.0297 * omega**2
    M = 0.8678 - 0.1785 * omega + 0.1401 * omega**2
    N = 2
    
    return L, M, N

@memoize
def a_and_b(gas, T):
    """Calculate the a and b intermediate parameters in the Peng-Robinson forumula 
    a : attraction parameter
    b : repulsion parameter
    
    Assume  temperature of 25 C if temp not given
    """
    # Reduced temperature and pressure
    Tc = gas_data[gas]['Tc']
    Pc = gas_data[gas]['Pc']

    Tr = T / Tc
    omega = gas_data[gas]['omega']
    
    
    # We do not use the L,M,N formulation as we have omega for
    # all our gases, and H2 just doesn't work with L,M,N at the pressures we use.
    if False:
        if 'L' in gas_data[gas]:
            L = gas_data[gas]['L']
            M = gas_data[gas]['M']
            N = gas_data[gas]['N']
        else:
            L, M, N = get_LMN(gas_data[gas]['omega'])            
        
        alpha1 = Tr ** (N*(M-1)) * np.exp(L*(1 - Tr**(M*N)))
    
    # updated wrt many compoudns, Pina-Martinez 2019:
    kappa = 0.3919 + 1.4996 * omega - 0.2721 * omega**2 + 0.1063 * omega**3
    
    
    # https://www.sciencedirect.com/science/article/abs/pii/S0378381218305041
    # 1978 Robinson and Peng
    if omega < 0.491: # omega for nC10, https://www.sciencedirect.com/science/article/abs/pii/S0378381205003493
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    else:
        kappa = 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.16666 * omega**3
        
    # Alpha function
    alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2

    # Coefficients for the cubic equation of state
    a = 0.45724 * (R * Tc)**2 / Pc * alpha
    b = 0.07780 * R * Tc / Pc

    return a, b

@memoize
def solve_for_Z(T, P, a, b):
   
    # Solve cubic equation for Z the compressibility
    A = a * P / (R * T)**2 # should have alpha in here? No..
    B = b * P / (R * T)
    
    c3 = 1
    c2 = -1 +B
    c1 = A - 3 * B**2 - 2 * B
    c0 = -A * B + B**2 + B**3

    # Solve the cubic equation for Z
    roots = np.roots([c3, c2, c1, c0])
 
    # Filter out complex roots and select the appropriate real root
    real_roots = roots[np.isreal(roots)].real
    Z = np.max(real_roots)  # Assuming vapor phase 
    
    return Z

@memoize
def peng_robinson(T, P, gas): # Peng-Robinson Equation of State
    if gas not in gas_mixtures:    
        a, b = a_and_b(gas, T)
    else:
        constants = z_mixture_rules(gas, T)
        a = constants[gas]['a_mix']
        b = constants[gas]['b_mix'] 
        
    Z = solve_for_Z(T, P, a, b)
    return Z


def viscosity_LGE(Mw, T_k, ϱ):
    """The  Lee, Gonzalez, and Eakin method, originally expressed in 'oilfield units'
    of degrees Rankine and density in g/cc, with a result in centiPoise
    doi.org/10.2118/1340-PA 1966
    Updated to SI: PetroWiki. (2023). 
    https://petrowiki.spe.org/Natural_gas_properties. 
    """

    T = T_k * 9/5 # convert Kelvins to Rankine
   
    # Constants for the Lee, Gonzalez, and Eakin #1
    k = (7.77 + 0.0063 * Mw) * T**1.5 / (122.4 + 12.9 * Mw + T)
    x = 2.57 + 1914.5 / T + 0.0095 * Mw # * np.exp(-0.025 * MWg) hallucination!
    y = 1.11 - 0.04 * x

    # Constants for the Lee, Gonzalez, and Eakin #2
    k = (9.4 + 0.02 * Mw) * T**1.5 / (209 + 19 * Mw + T)
    x = 3.5 + 986 / T + 0.01 * Mw
    y = 2.4 - 0.2 * x

    mu = 0.1 * k * np.exp(x * (ϱ / 1000)**y) #microPa.s

    return mu 

def print_bip():
    """Print out the binary interaction parameters
    """
    for g1 in gas_data:
        if g1 in k_ij:
            print("")
            for g2 in gas_data:
               if g2 in k_ij[g1]:
                pass
                print(f"{g1}:{g2} {k_ij[g1][g2]} - {estimate_k(g1,g2):.3f} {k_ij[g1][g2]/estimate_k(g1,g2):.3f}", end="\n")
            print("")

    for g1 in gas_data:
        for g2 in gas_data:
           pass
           print(f"{g1}:{g2}  {estimate_k(g1,g2):.3f}  ", end="")
        print("")

@memoize
def get_density(mix, p, T):
    if g in gas_data:
        ϱ_pg = p * gas_data[g]['Mw'] / (peng_robinson(T, p, g) * R * T)
        return ϱ_pg
        
    constants = z_mixture_rules(mix, T)
    a = constants[mix]['a_mix']
    b = constants[mix]['b_mix']
    Z_mix = solve_for_Z(T, p, a, b)
    mm = do_mm_rules(mix) # mean molar mass
    # For density, the averaging across the mixture (Mw) is done before the calc. of ϱ
    return p * mm / (Z_mix * R * T)

def get_Hc(g):
    """If the data is there, return the standard heat of combustion, 
    but in MJ/m³ not MJ/mol
    Uses molar volume at (15 degrees C, 1 atm) even though reference T for Hc is 25 C"""
    if g in gas_mixture_properties and 'Hc' in gas_mixture_properties[g]:
        hc = gas_mixture_properties[g]['Hc']
    elif g in gas_data and 'Hc' in gas_data[g]:
        hc = gas_data[g]['Hc']
    else:
        hc = 0
        composition = gas_mixtures[g]
        for pure_gas, x in composition.items():
            # Linear mixing rule for volume factor
            hc += x * gas_data[pure_gas]['Hc']
    # hc is in MJ/mol, so we need to divide by the molar volume at  (25 degrees C, 1 atm) in m³
    Mw = do_mm_rules(g)/1000 # Mw now in kg/mol not g/mol
    ϱ_0 = get_density(g, Atm, T15C) # in kg/m³
    molar_volume = Mw / ϱ_0  # in m³/mol
    
    if hc:
        return molar_volume, hc/molar_volume, hc
    else:
        return molar_volume, None, None
    
def print_density(g, p, T):
    ϱ = get_density(g, p, T)
    mm = do_mm_rules(g) # mean molar mass
    print(f"{g:15} {mm:6.3f}  {ϱ:.5f} ")
 
def print_wobbe(g):
    """HHV and Wobbe much be in MJ/m³, but at 15 C and 1 atm, not p and T as given
    UK NTS WObbe limits from     https://www.nationalgas.com/data-and-operations/quality
    
    "gas that is permitted in gas networks in Great Britain must have a relative density of ≤0.700"
    https://www.hse.gov.uk/gas/gas-safety-management-regulation-changes.htm
    also, relative density must be >0.7 and CO2 less than 2.5 mol.%
    """
    too_light = ""
    best = (47.20 + 51.41) / 2 # wobbe limits
    lowest = 47.20
    highest = 51.41
    width = 51.41 - 47.20
    
    # Wobbe is at STP: 15 C and 1 atm
    ϱ_0 = get_density(g, Atm, T15C)
    ϱ_air = get_density('Air', Atm, T15C)
    
    relative_ϱ = (ϱ_0/ϱ_air)
 
    wobbe_factor_ϱ = 1/np.sqrt(ϱ_0/ϱ_air)
    
    mv, hcmv, hc = get_Hc(g) 

    if relative_ϱ > 0.7:
        too_light = f"Rϱ > 0.7 ({relative_ϱ:.3f} = {ϱ_0:.3f} kg/m³)"
        
    # yeah, yeah: 'f' strings are great
    if hc:
        w = wobbe_factor_ϱ * hcmv
        niceness = 100*(w - best)/width  # 100*w/best - 100
        flag = f"{'nice':^8} {niceness:+.1f} %"
        if w < lowest:
            flag = f"{'LOW':^8}"
        if w  > highest:
            flag = f"{'HIGH':^8}"

        w = f"{w:>.5f}"
        hc = f"{hc:^11.3f}"
        hcmv = f"{hcmv:^11.4f}"
    else:
        w = f"{'-':^10}"
        hc = f"{'-':^11}"
        hcmv = f"{'-':^11}"
        flag = f"{'  -            '}"
   
    
    print(f"{g:15} {hc} {mv:.7f} {hcmv}{wobbe_factor_ϱ:>11.5f}   {w} {flag} {too_light}")
    
# ---------- ----------main program starts here---------- ------------- #

program = sys.argv[0]
stem = str(pl.Path(program).with_suffix(""))
fn={}
for s in ["z", "ϱ", "mu"]:
    f = stem  + "_" + s
    fn[s] = pl.Path(f).with_suffix(".png") 

for mix in gas_mixtures:
    composition = gas_mixtures[mix]
    check_composition(mix, composition)

    
# print_bip() # binary interaction parameters

dp = 47.5
tp = 15 # C
pressure =  Atm + dp/1000 # 1atm + 47.5 mbar, halfway between 20 mbar and 75 mbar
T15C = T273 + tp # K

# Print the densities at 15 C  - - - - - - - - - - -

print(f"\nDensity of gas at (kg/m³)at T={tp:.1f}°C and P={dp:.1f} mbar above 1 atm, i.e. P={pressure:.5f} bar")

gases = []
for g in gas_mixtures:
    gases.append(g)
for g in ["H2", "CH4"]:
    gases.append(g)

print(f"{'gas':13}{'Mw(g/mol)':6}  {'ϱ(kg/m³)':5} ")
for g in gases:
    print_density(g, pressure, T15C)
    
print(f"\nHc etc. all at 15°C and 1 atm = {Atm} bar. Wobbe limit is  47.20 to 51.41 MJ/m³")
print(f"W_factor_ϱ =  1/(sqrt(ϱ/ϱ(air))) ")

print(f"{'gas':13} {'Hc(MJ/mol)':11} {'MV₀(m³/mol)':11} {'Hc(MJ/m³)':11}{'W_factor_ϱ':11} Wobbe(MJ/m³) ")
for g in gases:
    print_wobbe(g)
 
print("'nice' values range from -50% to +50% from the centre of the valid Wobbe range.")
# Plot the compressibility  - - - - - - - - - - -

# Calculate Z0 for each gas
Z0 = {}
for gas in gas_data:
    Z0[gas] = peng_robinson(T273+25, pressure, gas)

# Plot Z compressibility factor for pure hydrogen and natural gases
temperatures = np.linspace(233.15, 323.15, 100)  
  # bar

plt.figure(figsize=(10, 6))

# Plot for pure hydrogen
Z_H2 = [peng_robinson(T, pressure, 'H2') for T in temperatures]
plt.plot(temperatures - T273, Z_H2, label='Pure hydrogen', linestyle='dashed')

    
# Plot for pure methane
Z_CH4 = [peng_robinson(T, pressure, 'CH4') for T in temperatures]
plt.plot(temperatures - T273, Z_CH4, label='Pure methane', linestyle='dashed')


# Plot for natural gas compositions. Now using correct temperature dependence of 'a'
ϱ_ng = {}
μ_ng = {}

for mix in gas_mixtures:
    mm = do_mm_rules(mix) # mean molar mass
    ϱ_ng[mix] = []
    μ_ng[mix] = []

    Z_ng = []
    for T in temperatures:
        # for Z, the averaging across the mixture (a, b) is done before the calc. of Z
        constants = z_mixture_rules(mix, T)
        a = constants[mix]['a_mix']
        b = constants[mix]['b_mix']
        Z_mix = solve_for_Z(T, pressure, a, b)
        Z_ng.append(Z_mix)
        
        # For density, the averaging across the mixture (Mw) is done before the calc. of ϱ
        ϱ_mix = pressure * mm / (Z_mix * R * T)
        ϱ_ng[mix].append(ϱ_mix)

        # for LGE viscosity, the averaging across the mixture (Mw) is done before the calc. of ϱ
        μ_mix = viscosity_LGE(mm, T, ϱ_mix)
        μ_ng[mix].append(μ_mix)

    if mix == "Air":
        continue 
    plt.plot(temperatures - T273, Z_ng, label=mix)

plt.title(f'Z  Compressibility Factor vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Z Compressibility Factor')
plt.legend()
plt.grid(True)

plt.savefig(fn["z"])
plt.close()

# Viscosity plot  LGE - - - - - - - - - - -

# Vicosity for gas mixtures LGE
# μ_ng[mix] was calculated earlier, but not plotted earlier
for mix in gas_mixtures:
   plt.plot(temperatures - T273, μ_ng[mix], label=mix)

# Viscosity plots for pure gases
μ_pg = {}
for g in ["H2", "CH4", "N2"]:
    μ_pg[g] = []
    mm_g = gas_data[g]['Mw']
    for T in temperatures:
        ϱ_g= pressure * mm / (peng_robinson(T, pressure, g) * R * T)
        μ = viscosity_LGE(mm_g, T, ϱ_g)
        μ_pg[g].append(μ)
    plt.plot(temperatures - T273, μ_pg[g], label= "pure " + g, linestyle='dashed')


plt.title(f'Dynamic Viscosity [LGE] vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Dynamic Viscosity (μPa.s) - THIS IS ALL WRONG, mistake in LGE formula')
plt.legend()
plt.grid(True)

plt.savefig("peng_mu_LGE.png")
plt.close()

# Viscosity plot  EXPTL values at 298K - - - - - - - - - - -

P = pressure
μ_g = {}
for mix in gas_mixtures:
    μ_g[mix] = []
    for T in temperatures:
        values = viscosity_values(mix, T, P)
        μ = linear_mix_rule(mix, values)
        # μ = hernzip_mix_rule(mix, values) # Makes no visible difference wrt to linear!
        #μ = explog_mix_rule(mix, values) # very slight change by eye
        μ_g[mix].append(μ)
    plt.plot(temperatures - T273, μ_g[mix], label= mix)
  

# Viscosity plots for pure gases
P = pressure
μ_pg = {}
for g in ["H2", "CH4", "N2", "O2"]:
    μ_pg[g] = []
    #vs, t = gas_data[g]['Vs']
    for T in temperatures:
        μ = viscosity_actual(g, T, P)
        μ_pg[g].append(μ)
    plt.plot(temperatures - T273, μ_pg[g], label= "pure " + g, linestyle='dashed')


plt.title(f'Dynamic Viscosity [data] vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Dynamic Viscosity (μPa.s) - linear Mw mixing rule')
plt.legend()
plt.grid(True)

plt.savefig(fn["mu"])
plt.close()

# ϱ/Viscosity plot Kinematic EXPTL values at 298K - - - - - - - - - - -

P = pressure
re_g = {}
for mix in gas_mixtures:
    re_g[mix] = []
    for i in range(len(μ_g[mix])):
        re_g[mix].append( ϱ_ng[mix][i] / μ_g[mix][i])
    plt.plot(temperatures - T273, re_g[mix], label= mix)
  

# Viscosity plots for pure gases 
P = pressure
re_pg = {}
for g in ["H2", "CH4", "N2", "O2"]: 
    re_pg[g] = []
    for T in temperatures:
        re = density_actual(g, T, P) / viscosity_actual(g, T, P)
        re_pg[g].append(re)
    plt.plot(temperatures - T273, re_pg[g], label= "pure " + g, linestyle='dashed')


plt.title(f'Density / Dynamic Viscosity vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Density/ Dynamic Viscosity (kg/m³)/(μPa.s) ')
plt.legend()
plt.grid(True)

plt.savefig("peng_re.png")
plt.close()

# Density plot  - - - - - - - - - - -

# pure gases
for g in ["H2", "CH4", "N2", "O2"]: 
    ϱ_pg = [pressure * gas_data[g]['Mw'] / (peng_robinson(T, pressure, g) * R * T)  for T in temperatures]
    plt.plot(temperatures - T273, ϱ_pg, label = "pure " + g, linestyle='dashed')

# Density plots for gas mixtures
for mix in gas_mixtures:
    plt.plot(temperatures - T273, ϱ_ng[mix], label=mix)

plt.title(f'Density vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Density (kg/m³)')
plt.legend()
plt.grid(True)

plt.savefig(fn["ϱ"])
plt.close()

# Plot the compressibility  as a function of Pressure - - - - - - - - - - -
T = T273+25

    
# Plot Z compressibility factor for pure hydrogen and natural gases
pressures = np.linspace(0, 80, 100)  # bar

plt.figure(figsize=(10, 6))

# Plot for pure hydrogen
Z_H2 = [peng_robinson(T, p, 'H2') for p in pressures]
plt.plot(pressures, Z_H2, label='Pure hydrogen', linestyle='dashed')

    
# Plot for pure methane
Z_CH4 = [peng_robinson(T, p, 'CH4') for  p in pressures]
plt.plot(pressures, Z_CH4, label='Pure methane', linestyle='dashed')

# Plot for pure ethane
Z_C2H6 = [peng_robinson(T, p, 'C2H6') for  p in pressures]
plt.plot(pressures, Z_C2H6, label='Pure ethane', linestyle='dashed')

# Plot for natural gas compositions. Now using correct temperature dependence of 'a'
ϱ_ng = {}
μ_ng = {}

for mix in gas_mixtures:
    mm = do_mm_rules(mix) # mean molar mass
    ϱ_ng[mix] = []
    μ_ng[mix] = []

    Z_ng = []
    for p in pressures:
        # for Z, the averaging across the mixture (a, b) is done before the calc. of Z
        constants = z_mixture_rules(mix, T)
        a = constants[mix]['a_mix']
        b = constants[mix]['b_mix']
        Z_mix = solve_for_Z(T, p, a, b)
        Z_ng.append(Z_mix)
        
        # For density, the averaging across the mixture (Mw) is done before the calc. of ϱ
        ϱ_mix = p * mm / (Z_mix * R * T)
        ϱ_ng[mix].append(ϱ_mix)

    # if mix == "Air":
        # continue 
    plt.plot(pressures , Z_ng, label=mix)

plt.title(f'Z  Compressibility Factor vs Pressure at {T} K')
plt.xlabel('Pressure (bar)')
plt.ylabel('Z Compressibility Factor')
plt.legend()
plt.grid(True)

plt.savefig("peng_z_p.png")
plt.close()
