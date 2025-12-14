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

        # epochs
        cfg.epochs = parser.getint('DEFAULT', 'epochs')

        # datasset path
        cfg.dataset_path = parser.get('DEFAULT', 'dataset_path')
        
        # optional scheduler settings
        # scheduler: none | step | cosine | plateau
        cfg.scheduler = parser.get('DEFAULT', 'scheduler', fallback='none')
        cfg.scheduler_step_per = parser.get('DEFAULT', 'scheduler_step_per', fallback='epoch')
        cfg.scheduler_step_size = parser.getint('DEFAULT', 'scheduler_step_size', fallback=10)
        cfg.scheduler_gamma = parser.getfloat('DEFAULT', 'scheduler_gamma', fallback=0.1)
        cfg.scheduler_T_max = parser.getint('DEFAULT', 'scheduler_T_max', fallback=50)
        cfg.scheduler_patience = parser.getint('DEFAULT', 'scheduler_patience', fallback=5)
        cfg.scheduler_warmup_steps = parser.getint('DEFAULT', 'scheduler_warmup_steps', fallback=0)

        # gradient clipping
        cfg.grad_clip = parser.getfloat('DEFAULT', 'grad_clip', fallback=1.0)

        return cfg