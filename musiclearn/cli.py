"""Command line interface for the musiclearn project."""
import click


@click.command()
def exp():
    """Run one experiment."""
    print("training...")


@click.group()
def cli():
    """Command line interface for the musiclearn project"""


def main():
    """"""
    cli.add_command(exp)
    cli()


if __name__ == "__main__":
    main()
