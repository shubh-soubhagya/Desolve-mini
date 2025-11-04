import git
import os

def clone_repo(repo_url, clone_dir):
    """
    Clone a GitHub repo into a given directory.
    
    Parameters:
        repo_url (str): GitHub repository URL.
        clone_dir (str): Directory path where repo should be cloned.
    """
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    target_path = os.path.join(clone_dir, repo_name)
    
    if os.path.exists(target_path):
        print(f"⚠️ Repo already exists at {target_path}")
    else:
        print(f"⏳ Cloning {repo_url} into {target_path} ...")
        git.Repo.clone_from(repo_url, target_path)
        print("✅ Clone complete!")

# Example usage
if __name__ == "__main__":
    repo_url = input("Enter GitHub Repo URL: ")
    clone_dir = r"cloned_repo"
    clone_repo(repo_url, clone_dir)
