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
            

## Usage 
- Refer to ```sahi_general_test.py``` for example usage 
