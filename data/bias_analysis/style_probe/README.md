# Style Probe
Investigation in to the accuracy of the Haiku LLM-as-Judge: does the judge grade articles off of actual bias or from perceived bias stemming from writing style?
We collect some sample articles not present in any of the corpora into two categories: 

1. Articles written in a confident / assertive style but are neutral. 
By the original interpretation of the rubric, these should score 0-1.
2. Articles written in a bland / boring style but contain partisan biases. 
By the original interpretation of the rubric, these should score >=2.

This directory is organized as follows:

| File                        | desc. |
|---                          | ---   |
| ```style_probe_final.csv``` | Final list of 35 articles, 18 for cat. 1 and 17 for cat. 2 |
| ```style_probe_grader_input.csv``` | csv passed to grader. categories, article titles, etc anything external that could be used to influence bias grade is removed|
| ```bias_scores_styleprobe.csv``` | haiku llm as judge grade |

Analysis of haiku grades is as follows:

| x      | Cat. 1    | Cat. 2             |
|---     |---        |---                 |
| N      | 18        | 17                 |
| Mean   | 0.39      | 1.82               |
| stddev | 0.50      | 1.01               |
| Range  | 0-1       | 0-3                |
| dist.  | 0x11, 1x7 | 0x2, 1x4, 2x6, 3x5 |

Category 2 havign a 1.43 higher mean bias shows that the llm as judge reliably scores partisan yet bland articles higher than neutral yet assertive ones.
Therefore, the judge scores based on substance rather than style in the articles. 


