## Overview

This document has the purpose of driving data investigation, informing modeling considerations, and serving as a reference point for the theory behind our modeling strategy. Due to this, information in this document may not be accurate and may not include adequate sources. 

## Introduction

Many factors will influnce the movement of Southern Resident Killer Whales (SRKW) in the Puget Sound and North-Eastern Pacific. For each of these factors, we will need to identify sufficient modeling methodologies so that we can capture their full predictive capacity.This document is to serve as a general overview of the many factors we have considered or plan to consider along with the reasoning behind their inclusion. We will structure this document into categories over which we can generalize modeling and organize feature development in an efficient and informed way.

## Factors

### 1. Prey

SRKW Orcas, as opposed to the larger mammal-eating Transient, base their diet entirely on fish, primarily Chinook salmon and a few other salmon types. Thus, SRKWs are driven to areas that are abundant in their primary food source. It is believed that the location of salmon waters, and perhaps salmon migratory patterns, is passed down over the generations. This means that we may be able to discern repeated "salmon-eating" routes which are re-visited over periods of low food availability and/or tied to temporal patterns that can be modeled.

#### i. Prey Distrubution

Overwhelmingly, Chinook salmon dominates their diet. It is clear that we must model the abunance of Chinook salmon, as well as other salmon in the Puget Sound and North-East Pacific to adequately model the occurance of SRKW orcas. It is hypothesized that SRKWs have a learned understanding of salmon movements in and out of the Puget Sound and base their movement on the movements of these fish. 

So, to accurately model SRKW orcas, it necessitates a modeling strategy on its own for salmon availability in the waters of the Puget Sound and North-East Pacific. 

We know the following about salmon:
- Salmon are released from hatcheries (man-made / non man-made) yearly. 
- Salmon swim up river to release eggs in a process known as "spawning" once a year. 
- Salmon can spend 1-5 years in the open ocean before returning to spawn. 
- Predation of salmon comes from varied sources including humans through recreational fishing and commercial fishing, sea lions, seals, sharks, transients?, bears to some effect, etc..
- We can model salmon prey availability from environmental factors. 

##### a. Predation-Model

We need to understand the movements of salmon and their availability through out the Puget Sound and North-East Pacific. Salmon have migratory patterns which we may be able to effectively model using timeseries regression that is spatially-constrained to areas of interest that are monitored via an array of measurement methods. 

Ideally, we will have salmon availability estimates approximated to our 10km hexagons as that will better align with our other modeling strategy. However, it may be sufficient to model salmon at a more general level to suggest that Orcas will be present/not-present. We could use the Marine Zones as defined by WDFW as the catch card and creel report data will align with these. Or, we could be more surgical by including weights obtained from the individual ramps and access points that make up the creel reports. This may make the salmon "abundance predictions" more precise in time and space but at this point, it is unclear if this will produce better results for SRKW presense prediction - other factors may influence the presence/non-presence of SRKWs more directly, such as large vessels, tides, etc..

These are some data source on Salmon predation: 
- PREDATION: SRKWs have competion with other predators. We will need to consider how to include the relative effect of all of these factors on prey abundance. 
    - <u><b> Recreational Fishing </b></u>
        - <b> Creel Reports </b> - released nightly from a variety of locations. This source will be observer-biased since the Department of Fish and Wildlife is not present at every catch location nor are they present for all catches. While we may lack spatial consistency, we may make up for in recency and long-history. The long-history may allow for sufficient interpolation at non-observed sites. 
        - <b> River Creel Reports </b> - these may provide context on Salmon migratory patterns up stream. Generally, the salmon that swim up river will no longer be available to SRKWs (except for in some specific generally unlikely cases).  
        - <b> Catch Card EOY Reports </b> - released yearly in July for the prior year April-April. These have recreational catch records for Marine areas and coastal regions. The report data only includes aggregated information at the monthly level but can provide insight into changes in human involvement (e.g., trends in number of people fishing). Generally, these reports may provide context for when humans are most actively targeting Salmon in the Puget Sound and river systems
        - <b> Catch Cards (Raw) </b> - these are not publically available but may be requested from DFW with a disclosure request. As the data may contain PII, we may need to put in a special request/work with the DFW to identify what can be requested. It is likely that these will provide much more context and information than the EOY reports. However, these may be include inaccuracte information or may not be complete/consistent.
        - <b> Dam Fish-Ladder Counts </b> - Counts over fish ladders may add context to salmon migratory patterns. Dams are typically far up river so these counts will spike when salmon abundance in the Sound is lower. 
        - <b> River Counts </b> - Other salmon count data exist for some of the river systems in Washington and Oregon by the general public or indigenous governments. We may be able to integrate these sources as well to better inform salmon migratory patterns.  

    - <u><b> Commercial Fishing </b></u>
        - <b> NOAA Creel Reports </b> - NOAA captures general salmon abundance in Federally-controlled waters off-shore in the North-East Pacific. Spikes in these reports may indicate salmon abunance off-shore and may provide insight to salmon migratory patterns and SRKW exits from the Puget Sound. 
        - <b> Fishery Job Openings </b> - (lagged variable)
        - ???

    - <u><b> Other Predator Abundance </b></u>
        - <b> Sea Lion Abundance </b> - We could look for time series data on sea lion counts. 
        - <b> Seal Abundance </b> - We could look for time series data on seal counts. 
        - <b> Porpoise Abundance </b> - Using the same whale sightings data sources (e.g., Acartia)
        - <b> Shark Abundance </b> - We could look at community reports
        - <b> Transient Orca Abundance </b> - May be loosely tied to seal / sea lion prey availability
    
##### b. Prey-Model

Many factors could influence the availability of prey for salmon. Environmental factors can influence the availability of prey which may drive salmon presence/non-presence given some lagging based on the level in the prey:predator hierarchy. Likewise, we may have information about prey abundance from catch reports and DFW data sources. 

Here are some sources we can look into: 
- PREY: Salmon have prey which can be identified or estimated based on environmental factors and human measurement.
    - <b><u> Environmental Factors </b></u>
        - <b> Chlorophyll-A </b> - Choloropyll content in the water column may be an indication of productivity and a lower level in the food hierarchy. Areas with high productivity may in turn have higher prey availability for salmon. We may be able to lag productivity over time and align to areas with high salmon abundance. Data for this can come from NOAA or NASA and may have a recency of 7-days depending on sensor platform. We may need to interpolate missing data over cloud periods.  
        - <b> Sea-Surface Temperature (SST_c) </b> - Salmon "like" areas within acceptable SST and may actively avoid areas over/under a certain threshold. We can measure SST and approximate deeped depths (if necessary). SST can be used to inform presence/avoidance. Data will come from NOAA or NASA. 
        - <b> Algae Blooms </b> - ??? need to look more into this and its effect on salmon abundance - hypothesize that salmon avoid areas with high algae content in the water. This can be caught with NDVI vs. NDWI and other methods with data obtained by Sentinel-2 or Landsat 8/9. 
        - <b> Upwelling </b> - upwelling often brings deep cold water and can bring other contaminents to the surface which may affect salmon presence/non-presence. Data will likely come from NOAA. 
        - <b> Salinity </b>
        - <b> Tide(s) </b> - Tides may influence where salmon are present depending on time of day. We can identify certain areas as no-salmon areas due to low tide. Data may come from NOAA.
        - <b> Depth </b> - Bathymetry data may give insight into predation, food abundance, and may relate to salmon migratory patterns. 
        - <b> Sunrise/Sunset Times </b>
        - <b> Surface Conditions </b> - local weather (wind speed, rain, snow)
        - <b> Flooding + Discharge </b>



### 2. Threats (Natural)

#### i. Predation

Killer Whales inhabit nearly every ocean and sea from the tropics to the poles. Across these diverse environments, Orcas are typically considered the primary apex predator. However, there are some natural threats which may impact Orca behaviour. In some cases, Orcas have avoidant behavior for certain species (especially consider Pilot Whales in the Atlantic). We will consider avoidant behaviour as a form of predation prevention from other animals.

It is said that SRKWs actively avoid Transient (Biggs) Killer Whales. It is not clear why they avoid the Transients but marine biologists are considering two potential reasons for this: cultural norms and threat. In either case, we will consider the arrival of transients to the Puget Sound as an avoidant factor which should negatively weight the presence of SRKWs given the presence of transients. 

#### ii. Weather

There are occasional beaching events which may be due to many factors. One factor to consider is the occurance of severe weather. Likewise, severe weather may impact an SRKW ability to hunt effectively as it may impact prey availability. 


### 3. Threats (Unnatural)



### 4. Competition




