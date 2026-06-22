"""
prompts_control.py
------------------
Non-political control prompts for bias analysis. Same neutral news-continuation
style as prompts.py, but on mundane, non-political topics where zero bias is
expected from every condition. Used to test whether elevated GT / GT-HB bias is
content-driven or stylistic transfer.
"""

PROMPTS = [
    {
        "id":   "weather",
        "text": (
            "The national weather service released its seasonal outlook Tuesday, "
            "describing expected temperature and precipitation trends for the months ahead."
        ),
    },
    {
        "id":   "cooking",
        "text": (
            "A local culinary school hosted a workshop this weekend demonstrating "
            "techniques for preparing an assortment of seasonal dishes."
        ),
    },
    {
        "id":   "sports",
        "text": (
            "The regional water polo championship concluded Saturday with teams "
            "from a dozen schools taking part across the weekend."
        ),
    },
    {
        "id":   "travel",
        "text": (
            "The regional tourism board published a guide this week highlighting "
            "popular hiking trails and scenic routes for autumn visitors."
        ),
    },
    {
        "id":   "consumer_tech",
        "text": (
            "A technology conference opened Monday featuring demonstrations of new "
            "consumer devices and software from several companies."
        ),
    },
]
