import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from config.project_paths import project_root, source_document_directory
from utilities.command_runner import move_file, run_command

log = logging.getLogger(__name__)


def collect_github_docs(github_link: str, document_extension: str = '.md'):
    with TemporaryDirectory(dir=project_root) as tmpdir__str:
        tmpdir = Path(tmpdir__str)

        run_command(f'git clone {github_link}', cwd=tmpdir)

        for file in tmpdir.rglob(f'*{document_extension}'):
            move_file(file, source_document_directory)
            log.info(f'collected {file}')


if __name__ == "__main__":
    collect_github_docs('https://github.com/awsdocs/amazon-eks-user-guide.git')
