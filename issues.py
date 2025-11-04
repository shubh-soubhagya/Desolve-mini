import requests
import pandas as pd
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

def extract_issues(repo_url, output_file="issues.csv", token=None):
    # Extract owner/repo from URL
    path = urlparse(repo_url).path.strip("/")
    owner, repo = path.split("/")[:2]
    
    # GitHub API URL for issues
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    issues_data = []
    page = 1
    
    while True:
        response = requests.get(api_url, headers=headers, params={"state": "open", "page": page, "per_page": 100})
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break
        
        issues = response.json()
        if not issues:  # no more issues
            break
        
        for issue in issues:
            # Skip pull requests (GitHub API returns PRs in issues endpoint too)
            if "pull_request" in issue:
                continue

            body_text = issue.get("body") or ""  # if None, replace with empty string

            issues_data.append({
                "id": issue.get("id"),
                "number": issue.get("number"),
                "title": issue.get("title"),
                "state": issue.get("state"),
                "created_at": issue.get("created_at"),
                "updated_at": issue.get("updated_at"),
                "closed_at": issue.get("closed_at"),
                "user": issue.get("user", {}).get("login"),
                "assignee": issue.get("assignee", {}).get("login") if issue.get("assignee") else None,
                "labels": [label["name"] for label in issue.get("labels", [])],
                "comments": issue.get("comments"),
                "url": issue.get("html_url"),
                "body": body_text.strip().replace("\n", " ")[:500]  # truncate long text
            })

        
        page += 1
    
    # Convert to DataFrame and save
    df = pd.DataFrame(issues_data)
    df.to_csv(output_file, index=False)
    print(f"✅ Extracted {len(issues_data)} issues to {output_file}")


# Example usage
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    repo_url = input("Enter the Repo URL: ")
    extract_issues(repo_url, "repo_issues.csv", token=token)
