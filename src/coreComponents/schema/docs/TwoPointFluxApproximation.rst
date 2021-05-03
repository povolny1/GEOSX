

========================= ============ ======== ========================================================== 
Name                      Type         Default  Description                                                
========================= ============ ======== ========================================================== 
areaRelTol                real64       1e-08    Relative tolerance for area calculations.                  
coefficientName           string       required Name of coefficient field                                  
fieldName                 string       required Name of primary solution field                             
mechanicalStabCoefficient real64       0        Value of the (currently uniform) stabilization coefficient 
name                      string       required A name is required for any non-unique nodes                
targetRegions             string_array {}       List of regions to build the stencil for                   
========================= ============ ======== ========================================================== 


