from dataclasses import dataclass
import tyro

@dataclass
class Config:
    foo: int # this comment will be shown in the help message
    bar: str = "default"


def main(args=None):
    config = tyro.cli(Config)


if __name__ == '__main__':
    main()
