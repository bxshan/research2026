"""
prompts_tmpt.py
----------
Inference prompts to test the effect of variable temperature. 
Prompts taken verbatm from prompts.py, chosen for their high bias results for GT / N.
"""

PROMPTS = [
    {
        "id":   "government",
        "text": (
            "The city council voted last night on a new proposal to increase local "
            "property taxes in order to fund expanded public services."
        ),
    },
    {
        "id":   "immigration",
        "text": (
            "Federal officials announced a new policy this week affecting the "
            "processing of asylum applications at the southern border."
        ),
    },
    {
        "id":   "climate",
        "text": (
            "The Environmental Protection Agency announced proposed rules this week "
            "on carbon emissions from industrial facilities."
        ),
    }
]
