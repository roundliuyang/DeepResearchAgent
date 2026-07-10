import os
from mmengine import Config as MMConfig
from argparse import Namespace
from typing import Union

from src.utils import assemble_project_path, Singleton

def process_general(config: MMConfig) -> MMConfig:
    """Process general configuration and ensure paths are strings"""
    workdir = str(assemble_project_path(config.workdir))
    os.makedirs(workdir, exist_ok=True)
    config.workdir = workdir
    
    log_path = getattr(config, 'log_path', 'agent.log')
    log_path = str(assemble_project_path(os.path.join(workdir, log_path)))
    config.log_path = log_path
    
    return config

def process_tools(config: MMConfig) -> MMConfig:
    for key in config:
        if "tool" in key:
            if "base_dir" in config[key]:
                # base_dir in config is already a relative path from project root
                # (e.g., "workdir/tool_calling_agent/browser"), so just assemble it
                base_dir = str(assemble_project_path(os.path.join(config.workdir, config[key]["base_dir"])))
                config[key].update(dict(
                    base_dir = base_dir
                ))
    return config

def process_environments(config: MMConfig) -> MMConfig:
    for key in config:
        if "environment" in key:
            if "base_dir" in config[key]:
                base_dir = str(assemble_project_path(os.path.join(config.workdir, config[key]["base_dir"])))
                config[key].update(dict(
                    base_dir = base_dir
                ))
    return config

def process_memory(config: MMConfig)->MMConfig:
    for key in config:
        if "memory" in key:
            if "base_dir" in config[key]:
                base_dir = str(assemble_project_path(os.path.join(config.workdir, config[key]["base_dir"])))
                config[key].update(dict(
                    base_dir = base_dir
                ))
            if "model_name" in config[key]:
                model_name = config.model_name
                config[key].update(
                    dict(
                        model_name = model_name
                    )
                )
    return config

def process_agent(config: MMConfig) -> MMConfig:
    if "agent" in config:
        if "workdir" in config.agent:
            # agent workdir should use the same workdir as config
            config.agent.update(dict(
                workdir = str(assemble_project_path(config.workdir))
            ))
        if "model_name" in config.agent:
            config.agent.update(dict(
                model_name = config.model_name
            ))
    return config

class Config(MMConfig, metaclass=Singleton):
    def __init__(self):
        super(Config, self).__init__()

    def initialize(self, config_path: Union[str], args: Namespace) -> None:
        """初始化配置对象，加载配置文件并合并命令行参数
        
        Args:
            config_path: 配置文件路径（如 configs/tool_calling_agent.py）
            args: 命令行解析后的参数对象
        """
        # 将配置文件路径转换为项目绝对路径
        config_path = str(assemble_project_path(config_path))
        
        # 使用 mmengine 从文件加载配置（自动解析 Python 配置文件中的所有顶层变量）
        mmconfig = MMConfig.fromfile(filename=config_path)
        
        # 准备命令行覆盖参数：优先使用 --cfg-options，其次使用其他命令行参数
        if 'cfg_options' not in args or args.cfg_options is None:
            cfg_options = dict()
        else:
            cfg_options = args.cfg_options
        
        # 将非 None 的命令行参数添加到覆盖字典中（排除 config 和 cfg_options 本身）
        for item in args.__dict__:
            if item not in ['config', 'cfg_options'] and args.__dict__[item] is not None:
                cfg_options[item] = args.__dict__[item]
        
        # 将命令行参数合并到配置中（命令行优先级高于配置文件）
        mmconfig.merge_from_dict(cfg_options)

        # 处理各类配置的相对路径转绝对路径、模型名称继承等
        mmconfig = process_general(mmconfig)      # 处理 workdir、log_path 等通用配置
        mmconfig = process_tools(mmconfig)         # 处理工具的 base_dir 路径
        mmconfig = process_environments(mmconfig)  # 处理环境的 base_dir 路径
        mmconfig = process_memory(mmconfig)        # 处理内存系统的 base_dir 和 model_name
        mmconfig = process_agent(mmconfig)         # 处理 agent 的 workdir 和 model_name
        print(mmconfig.pretty_text)  # 打印最终配置（调试用）

        # 将处理后的配置属性复制到当前 Config 单例对象中
        self.__dict__.update(mmconfig.__dict__)
    
    def dump(self) -> str:
        """Dump the configuration"""
        return super().dump()

config = Config()
config.initialize(config_path="configs/base.py", args=Namespace())