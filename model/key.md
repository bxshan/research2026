## Adapters
| File                                 | Model   | Platform    | Config | Steps | LoRA Rank | Final Loss | Training Time (hrs) | 
|---                                   |---      |---          |---     |---    |---        | ---        | ---                 |
| ```qwen-sft-gt```                    | qwen    | M5          | Local  | 100   | -         | -          | -                   |
| ```qwen-sft-ps```                    | qwen    | M5          | Local  | 100   | -         | -          | -                   |
| ```llama-sft-gt_20260419_023444```   | llama   | M5          | Local  | 1500  | 8         | 1.845546   | 6.16                |
| ```llama-sft-ps_20260419_085139```   | llama   | M5          | Local  | 1500  | 8         | 0.340333!  | 6.30                |
| ```llama-sft-gt_20260419_091436```   | llama   | Runpod L40S | Cloud  | 5000  | 64        | 1.855181   | 18.4                |
| ```llama-sft-ps_20260420_033833```   | llama   | Runpod L40S | Cloud  | 5000  | 64        | 0.572233!  | 16.8                |

note: 
- hyperparams are under ```./cfgs/```. 
- File names for the log csvs correspond to the adapters timestamp.
- PS SFT final losses indicate overfitting


## Inference Results
| File                                    | Model | Platform | Adapters Used |
|---                                      |---    |---       |---            |
| ```infer_results_20260413_230752.csv``` | qwen  | M5       | qwen-sft-*    |
| ```infer_results_20260418_175012.csv``` | qwen  | M5       | qwen-sft-*    |


