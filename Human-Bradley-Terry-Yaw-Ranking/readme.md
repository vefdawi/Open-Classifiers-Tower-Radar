# Pairwise Angle Comparison

Complementing the human Bradley-Terry-style annotation, large frontier vision-language models have been utilized in the same manner. Here, radagrams have been presented in the following fashion:

![Example radargram comparison](https://raw.githubusercontent.com/vefdawi/Open-Classifiers-Tower-Radar/main/Human-Bradley-Terry-Yaw-Ranking/VLM_Upscaled_Comparison_Example.png)

And subsequent VLM instructions read:

'''The image contains two radar scans placed side by side. The left panel is labeled 'A'; the right panel is labeled 'B'. Each panel shows a bright crescent shaped like the letter C, namely the trajectory of a flying object. The C has a backbone and two arms curving away from it. The arms do not touch, there is thus a void/space. The 'opening direction' is where this void points to. The opening direction faces either upward, to the right, or opens downward (and so on).

Please reason carefully, step by step! Step 1: Panel A: Where is backbone, where is the gap? Which direction does the C open to? Step 2: Panel B: Where is its backbone, where is the void? Which direction does the openspace point to? Step 3: Comparison: Relative to each other, which panel's c-shaped curve opens more toward the top of its image?

After your reasoning, end your response with exactly one of these two lines and nothing else... 

ANSWER: A
ANSWER: B'''
