# SAHI General


## Requirements
- SAHI :  0.10.5 <br>
  ``` pip install sahi ```
- CV Model : requires the below functions  
  - classname_to_idx : int <br>
  
        Parameters
        ----------
        classname : str
          class name of object
            
        Returns
        -------
        int
          index of the classname 
          
  - get_detections_dict :  <br>
  
        Parameters
        ----------
        frames : List[ndarray]
            list of input images
        classes : List[str], optional
            classes to focus on
        buffer_ratio : float, optional
            proportion of buffer around the width and height of the bounding box
            
        Returns
        -------
        List[dict]
            list of detections for each frame with keys: label, confidence, t, l, b, r, w, h
            
## Using sahi_general as a package for inference

- Clone this repository
- Make sure the requirements above are installed
- in the main project folder, install sahi_general as a package
```
python3 -m pip install --no-cache-dir /path/to/sahi_general
```
OR as an editable package (if you need to make changes to the code, faster to build)
```
python3 -m pip install -e /path/to/sahi_general
```
- import the sahi_general wrapper class for inference (refer to ```sahi_general_test.py``` for example usage )
```
from sahi_general.sahi_general import SahiGeneral
```
