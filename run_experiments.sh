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
#python musiclearn_cli.py fit-mtvae mtvae_0004 --beats-per-phrase 16 --lstm-units=256 --embedding-dim=16 --dropout-rate 0.4 --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0005 --beats-per-phrase 4 --lstm-units 512 --embedding-dim 0 --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0006 --beats-per-phrase 4 --lstm-units 256 --latent-dim 1024 --embedding-dim 128 --patience 100
#python musiclearn_cli.py fit-mtvae mtvae_0007 --beats-per-phrase 4 --lstm-units 512 --latent-dim 1024 --embedding-dim 32 --patience 200
#python musiclearn_cli.py fit-mtvae mtvae_0008 --beats-per-phrase 4 --lstm-units 512 --latent-dim 2048 --embedding-dim 32 --patience 100
#python musiclearn_cli.py fit-mtvae mtvae_0009 --beats-per-phrase 4 --lstm-units 128
#python musiclearn_cli.py fit-mtvae mtvae_0010 --beats-per-phrase 16 --lstm-units 128 --latent-dim 512 --embedding-dim 32  --dropout-rate 0.4 --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0011 --beats-per-phrase 16 --gru --lstm-units 128 --latent-dim 512 --embedding-dim 32  --dropout-rate 0.4 --patience 50
#python musiclearn_cli.py fit-mtvae mtvae_0012 --beats-per-phrase 16 --lstm-units 512 --latent-dim 2048 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtvae_0013 --beats-per-phrase 8 --lstm-units 128 --latent-dim 512 --dropout-rate 0.4 --patience 20
#python musiclearn_cli.py fit-mtvae mtvae_0014 --beats-per-phrase 8 --lstm-units 256 --latent-dim 512 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtvae_0015 --beats-per-phrase 8 --lstm-units 128 --latent-dim 128 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtvae_0016 --beats-per-phrase 4 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtvae_0017 --beats-per-phrase 4 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --bidirectional --patience 20
#python musiclearn_cli.py fit-mtvae mtvae_0018 --beats-per-phrase 8 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0001 --beats-per-phrase 4 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0002 --beats-per-phrase 4 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --learning-rate 0.001 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0003 --beats-per-phrase 4 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20 --batch-size 96
#python musiclearn_cli.py fit-mtvae mtsvae_0004 --beats-per-phrase 4 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20 --batch-size 64
#python musiclearn_cli.py fit-mtvae mtsvae_0005 --beats-per-phrase 8 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0006 --beats-per-phrase 4 --lstm-units 128 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20 --gru
#python musiclearn_cli.py fit-mtvae mtsvae_0007 --beats-per-phrase 4 --lstm-units 64 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0008 --beats-per-phrase 4 --lstm-units 256 --latent-dim 128 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0009 --beats-per-phrase 4 --lstm-units 128 --latent-dim 256 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0010 --beats-per-phrase 4 --lstm-units 128 --latent-dim 512 --embedding-dim 8 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0011 --beats-per-phrase 4 --lstm-units 128 --latent-dim 512 --embedding-dim 4 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0012 --beats-per-phrase 4 --lstm-units 128 --latent-dim 512 --embedding-dim 16 --dropout-rate 0.5 --patience 20
#python musiclearn_cli.py fit-mtvae mtsvae_0013 --beats-per-phrase 4 --lstm-units 128 --latent-dim 512 --embedding-dim 32 --dropout-rate 0.5 --patience 20
python musiclearn_cli.py fit-mtvae mtsvae_0014 --beats-per-phrase 4 --lstm-units 128 --latent-dim 512 --embedding-dim 8 --dropout-rate 0.5 --patience 20 --bidirectional
