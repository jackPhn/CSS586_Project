#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 4 --lstm-units 128
#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 4 --dropout-rate 0.5
#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 8
#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 8 --lstm-units=512
#python musiclearn_cli.py fit-mtvae mtvae_0001 --beats-per-phrase 8 --lstm-units=512 --dropout-rate 0.2
#python musiclearn_cli.py fit-mtvae mtvae_0002 --beats-per-phrase 16 --lstm-units=256 --embedding-dim=16 --dropout-rate 0.4 --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0003 --beats-per-phrase 8 --gru --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0003 --beats-per-phrase 8 --gru --embedding-dim=32 --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0003 --beats-per-phrase 16 --gru --embedding-dim=32 --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0003 --beats-per-phrase 16 --gru --embedding-dim=48 --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0003 --beats-per-phrase 16 --lstm --embedding-dim=48 --patience 50
python musiclearn_cli.py fit-mtvae mtvae_0004 --beats-per-phrase 16 --lstm-units=256 --embedding-dim=16 --dropout-rate 0.4 --patience 50
