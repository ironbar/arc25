from dataclasses import dataclass
import tyro

@dataclass
class Config:
    filepath: str = '../docs/05_Solution_Summary.md'
    output_filepath: str = "kaggle.md"


def main(args=None):
    config = tyro.cli(Config)
    with open(config.filepath, 'r') as f:
        content = f.read()
    # replace image urls
    content = content.replace("../modeling", "https://ironbar.github.io/arc25/modeling")
    content = content.replace("(res/", "(https://ironbar.github.io/arc25/res/")
    content = content.replace("(modeling/res/", "(https://ironbar.github.io/arc25/modeling/res/")
    # TODO: replace admonitions
    with open(config.output_filepath, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    main()
