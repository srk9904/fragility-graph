import os
import subprocess
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class GitService:
    def __init__(self, repo_path):
        self.repo_path = repo_path

    def get_co_change_patterns(self, limit=100):
        """
        Analyzes git history to find files that frequently change together.
        Returns a list of tuples: ((file_a, file_b), frequency)
        """
        try:
            # Get the list of changed files per commit for the last N commits
            cmd = ["git", "log", f"-n {limit}", "--pretty=format:%H", "--name-only"]
            output = subprocess.check_output(cmd, cwd=self.repo_path, text=True)
            
            commits = []
            current_files = []
            
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    if current_files:
                        commits.append(current_files)
                        current_files = []
                    continue
                
                # Check if it's a hash (simple check)
                if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                    if current_files:
                        commits.append(current_files)
                        current_files = []
                else:
                    # It's a file path
                    current_files.append(line)
            
            # Count co-occurrences
            pair_counts = Counter()
            for file_list in commits:
                file_list = sorted(list(set(file_list))) # Unique and sorted to avoid double counting (A,B) and (B,A)
                for i in range(len(file_list)):
                    for j in range(i + 1, len(file_list)):
                        pair_counts[(file_list[i], file_list[j])] += 1
            
            # Filter patterns (e.g., at least 2 co-changes)
            patterns = [ (pair, count) for pair, count in pair_counts.items() if count >= 2 ]
            return sorted(patterns, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error(f"Git mining failed: {e}")
            return []

# Placeholder for integration
git_service = GitService(os.getcwd())
