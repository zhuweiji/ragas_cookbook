import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm import tqdm

from config.project_paths import project_root, source_document_directory
from utilities.command_runner import move_file, run_command

log = logging.getLogger(__name__)


def collect_github_docs(github_link: str, document_extension: str = '.md'):
    directory_name = github_link.split('/')[-1].replace('.git', '')

    with TemporaryDirectory(dir=project_root) as tmpdir__str:
        tmpdir = Path(tmpdir__str)

        run_command(f'git clone {github_link}', cwd=tmpdir)

        for file in tqdm(list(tmpdir.rglob(f'*{document_extension}'))):
            move_file(file, source_document_directory / directory_name)
            log.info(f'collected {file}')


if __name__ == "__main__":
    # collect_github_docs('https://github.com/awsdocs/amazon-eks-user-guide.git')
    # collect_github_docs('https://github.com/microsoft/vscode-docs.git')
    # collect_github_docs('https://github.com/FrameworkComputer/linux-docs.git')
    pass
