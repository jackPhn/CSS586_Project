#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 4 --lstm-units 128
#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 4 --dropout-rate 0.5
#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 8
#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 8 --lstm-units=512
python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 8 --lstm-units=512 --dropout-rate 0.2
