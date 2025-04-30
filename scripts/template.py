from dataclasses import dataclass
import tyro

@dataclass
class Config:
    foo: int # The answer to life, the universe, and everything
    bar: str = "default"


def main(args=None):
    config = tyro.cli(Config)


if __name__ == '__main__':
    main()
