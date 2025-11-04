import os
import pandas as pd

def extract_files_to_csv(repo_dir, output_csv="repo_files.csv"):
    """
    Extracts all file contents from a cloned repo directory (recursively) 
    and saves them to a CSV with file_name, file_content, file_path, file_extension.
    Skips image/video/binary files and ignores .git/ directories.
    """
    file_data = []
    license_found = False

    skip_extensions = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".ico", ".svg",
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm",
        ".mp3", ".wav", ".flac", ".ogg", ".aac",
        ".exe", ".dll", ".so", ".bin", ".dat", ".zip", ".tar", ".gz"
    }

    skip_dirs = {".git", "__pycache__", ".idea", ".vscode", "node_modules"}

    for root, dirs, files in os.walk(repo_dir):
        # modify dirs in-place to skip unwanted directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

        for file in files:
            file_path = os.path.join(root, file)
            file_name = file
            file_extension = os.path.splitext(file)[1].lower()

            # Detect LICENSE (case insensitive)
            if file_name.lower().startswith("license"):
                license_found = True

            if file_extension in skip_extensions:
                continue  # skip binary/image/video files

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                content = f"Error reading file: {e}"

            file_data.append({
                "file_name": file_name,
                "file_content": content,
                "file_path": file_path,
                "file_extension": file_extension
            })

    df = pd.DataFrame(file_data)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    if license_found:
        print(f"✅ Extracted {len(file_data)} files into {output_csv} (LICENSE found)")
    else:
        print(f"✅ Extracted {len(file_data)} files into {output_csv} (⚠️ NO LICENSE found)")


# Example usage
if __name__ == "__main__":
    repo_dir = r"cloned_repo"
    extract_files_to_csv(repo_dir, "repo_files_data.csv")
