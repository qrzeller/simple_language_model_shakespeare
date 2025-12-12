import configparser

class Config:
    N: int  # context length or max sequence length
    @staticmethod
    def load_from_file(filepath: str) -> "Config":
        parser = configparser.ConfigParser()
        parser.read(filepath)

        cfg = Config()
        # required values (will raise if missing)
        cfg.block_size = parser.getint('DEFAULT', 'block_size')
        cfg.N = cfg.block_size  # alias used elsewhere in the code

        cfg.model_dim = parser.getint('DEFAULT', 'n_embd')
        cfg.num_heads = parser.getint('DEFAULT', 'n_head')
        cfg.num_layers = parser.getint('DEFAULT', 'n_layer')
        cfg.dropout = parser.getfloat('DEFAULT', 'dropout')
        cfg.batch_size = parser.getint('DEFAULT', 'batch_size')
        cfg.learning_rate = parser.getfloat('DEFAULT', 'learning_rate')
        cfg.device = parser.get('DEFAULT', 'device')

        # to check if text is compatible with model
        cfg.vocab_size = parser.getint('DEFAULT', 'vocab_size')

        return cfg