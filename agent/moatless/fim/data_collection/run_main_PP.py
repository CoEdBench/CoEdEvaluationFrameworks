import subprocess

# Define parameter list, each parameter group is a dictionary
params_list = [
    # {
    #     "repo": r"D:\Data\2025\repos\Python\AutoGPT",
    #     "repo_name": "AutoGPT",
    #     "output": r"D:\Data\2025\single_point\single_coarse_autogpt.jsonl",
    #     "limit": 100
    # },
    # {
    #     "repo": r"D:\Data\2025\repos\Python\scikit-learn",
    #     "repo_name": "scikit-learn",
    #     "output": r"D:\Data\2025\single_point\single_coarse_scikit-learn.jsonl",
    #     "limit": 100
    # },
    # # Add other parameter groups
    # {
    #     "repo": r"D:\Data\2025\repos\TS\openclaw",
    #     "repo_name": "openclaw",
    #     "output": r"D:\Data\2025\single_point\single_coarse_openclaw.jsonl",
    #     "limit": 100
    # },
    # {
    #     "repo": r"D:\Data\2025\repos\TS\n8n",
    #     "repo_name": "n8n",
    #     "output": r"D:\Data\2025\single_point\single_coarse_n8n.jsonl",
    #     "limit": 100
    # },
    {
        "repo": r"D:\Data\2025\repos\Java\elasticsearch",
        "repo_name": "elasticsearch",
        "output": r"D:\Data\2025\single_point\single_coarse_elasticsearch2.jsonl",
        "limit": 100
    },
    {
        "repo": r"D:\Data\2025\repos\Java\spring-boot",
        "repo_name": "spring-boot",
        "output": r"D:\Data\2025\single_point\single_coarse_spring-boot.jsonl",
        "limit": 100
    },
]

# Run each group of parameters serially
for params in params_list:
    command = [
        "python",
        r"D:/Data/pythonCodes/nep_builder/main_p1_collect_commits.py",
        "--repo", params["repo"],
        "--repo_name", params["repo_name"],
        "--output", params["output"],
        "--limit", str(params["limit"])
    ]

    # Call subprocess to execute the command
    subprocess.run(command, check=True)
