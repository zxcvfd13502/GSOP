import importlib
import os
import yaml
import sys
import json

import traceback
import argparse
import numpy as np
import datetime
import pdb
# import yaml
from easydict import EasyDict

import warnings
import traceback

warnings.simplefilter("ignore", category=DeprecationWarning)

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from pathlib import Path
from typing import Union
import hashlib

from lmms_eval import evaluator, utils
from lmms_eval.tasks import initialize_tasks, include_path, get_task_dict
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.logging_utils import WandbLogger
from loguru import logger as eval_logger


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument("--batch_size", type=str, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--show_task_to_terminal",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles",
    )
    # args added for GSOP Inference

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="The default attention implementation to use. Options are 'eager', 'flash_attention_2'. ",
    )

    parser.add_argument("--desire_prune_ratio", type = float, default = 0.5)

    parser.add_argument(
        "--prune_config_path",
        type=str,
        default=None,
        help="Path to the config file of the pruning policy.",
    )

    # args added for GSOP Search
    parser.add_argument(
        "--search_config_path",
        type=str,
        default=None,
        help="Path to the config file of the pruning search.",
    )

    parser.add_argument("--acc_threshold_ratio", type = float, default = 0.3)
    parser.add_argument("--greedy_steps", type = int, default = 15)
    parser.add_argument("--greedy_task", type = str, default = 'gqa')
    parser.add_argument("--greedy_metric", type = str, default = 'exact_match,none')
    parser.add_argument("--img_token_grouping", type = str, default = 'cls')
    parser.add_argument('--time_limit', type=float, default = -1)
    parser.add_argument('--key_img_ratio', type=float, default = 0.25)
    parser.add_argument("--prune_groups", type = int, nargs="+", default=[1, 2])
    parser.add_argument("--prune_ops", type = int, nargs="+", default=[1, 2, 0])
    parser.add_argument("--mid_out_dir", type = str, default = '/projectnb/ivc-ml/amliu/slm/lmms-eval/stamp_mid_res')
    parser.add_argument("--greedy_exp_name", type = str, default = 'l157_greedy')
    parser.add_argument("--selection_seed", type = int, default=None)
    parser.add_argument('--remove_key_out', default=False, action='store_true', help='use half of the layers for lossless')
    parser.add_argument("--remove_key_layer", type=int, nargs="+", default=None)
    parser.add_argument('--ll_half', default=False, action='store_true', help='use half of the layers for lossless')
    args = parser.parse_args()
    return args

def parse_prune_args(args: argparse.Namespace) -> None:
    if args.prune_config_path:
        yaml_path = os.path.join(args.prune_config_path, "configs.yaml")
        # with open(yaml_path, "r") as f:
        #     prune_config = yaml.safe_load(f)
        with open(yaml_path, "r") as f:
            prune_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        prune_config = EasyDict(prune_config)
        prune_config.num_sys_token = prune_config.token_nums[0]
        prune_config.num_img_token = prune_config.token_nums[1] + prune_config.token_nums[2]
        prune_config.num_txt_token = prune_config.token_nums[3]
        prune_config.remove_key_layer = args.remove_key_layer
        prune_config.prune_config_path = args.prune_config_path
        prune_config.desire_prune_ratio = args.desire_prune_ratio
    else:
        prune_config = None
    return prune_config

def parse_search_args(args: argparse.Namespace) -> None:
    if args.search_config_path:
        with open(args.search_config_path, "r") as f:
            search_config = yaml.safe_load(f)
        search_config = EasyDict(search_config)
        search_config.img_group_ratios = [args.key_img_ratio, 1 - args.key_img_ratio]
        search_config.token_groups = ['sys', 'img0', 'img1', 'txt']
        search_config.random_vtokens = args.img_token_grouping == "random"
        search_config.img_cls_vtokens = args.img_token_grouping == "cls"
        search_config.acc_threshold_ratio = args.acc_threshold_ratio
        search_config.greedy_steps = args.greedy_steps
        search_config.greedy_task = args.greedy_task
        search_config.greedy_metric = args.greedy_metric
        search_config.prune_groups = args.prune_groups
        search_config.prune_ops = args.prune_ops
        search_config.ll_half = args.ll_half
        search_config.mid_out_dir = args.mid_out_dir
        search_config.greedy_exp_name = args.greedy_exp_name
        search_config.selection_seed = args.selection_seed
        search_config.remove_key_out = args.remove_key_out
        search_config.remove_key_layer = args.remove_key_layer
        search_config.token_nums = [search_config.num_sys_token, 0, 0, search_config.num_txt_token]
        search_config.token_nums[1] = int(search_config.num_img_token * search_config.img_group_ratios[0])
        search_config.token_nums[2] = search_config.num_img_token - search_config.token_nums[1]
    else:
        search_config = None
    return search_config

def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_eval_args()

    # Check if no arguments were passed after parsing
    if len(sys.argv) == 1:
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                          │")
        print("│ `lmms-eval --model llava --model_path liuhaotian/llava-v1.6-7b --tasks okvqa` │")
        print("│ Use `lmms-eval --help` for more information.                                  │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    # reset logger
    eval_logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args_list = []
    results_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # initialize Accelerator
    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
    if accelerator.is_main_process:
        is_main_process = True
    else:
        is_main_process = False

    for args in args_list:
        try:
            if is_main_process and args.wandb_args:  # thoughtfully we should only init wandb once, instead of multiple ranks to avoid network traffics and unwanted behaviors.
                wandb_logger = WandbLogger(args)

            results, samples = cli_evaluate_single(args)
            results_list.append(results)

            accelerator.wait_for_everyone()
            if is_main_process and args.wandb_args:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.wandb_log_samples and samples is not None:
                    wandb_logger.log_eval_samples(samples)

                wandb_logger.finish()

        except Exception as e:
            traceback.print_exc()
            eval_logger.error(f"Error during evaluation: {e}")
            traceback.print_exc()
            results_list.append(None)

    for args, results in zip(args_list, results_list):
        # cli_evaluate will return none if the process is not the main process (rank 0)
        if results is not None:
            print_results(args, results)


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    initialize_tasks(args.verbosity)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")
    if args.limit:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")
    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
            eval_logger.info(f"Including path: {args.include_path}")
            include_path(package_tasks_location)

    if args.tasks is None:
        task_names = ALL_TASKS
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format(f"\n - ".join(sorted(ALL_TASKS))))
        sys.exit()
    elif args.tasks == "list_with_num":
        log_message = (
            "\n" + "=" * 70 + "\n" + "\n\tYou are trying to check all the numbers in each task." + "\n\tThis action will download the complete dataset." + "\n\tIf the results are not clear initially, call this again." + "\n\n" + "=" * 70
        )
        eval_logger.info(log_message)
        for task_name in sorted(ALL_TASKS):
            try:
                task_dict = get_task_dict([task_name], model_name="llava")
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                eval_logger.info(f"\nTask : {task_obj.config.task}\n - #num : {len(task_obj.test_docs()) if task_obj.has_test_docs() else len(task_obj.validation_docs())}")
            except Exception as e:
                eval_logger.debug(f"\nTask : {task_name} fail to load \n Exception : \n {e}")
        sys.exit()
    else:
        tasks_list = args.tasks.split(",")
        eval_logger.info(f"Evaluating on {len(tasks_list)} tasks.")
        task_names = utils.pattern_match(tasks_list, ALL_TASKS)
        task_missing = [task for task in tasks_list if task not in task_names and "*" not in task]  # we don't want errors if a wildcard ("*") task name was used

        if task_missing:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}. Try `lmms-eval --tasks list` for list of available tasks",
            )
            # eval_logger.warn(f"Tasks {missing} were not found. Try `lmms-eval --tasks list` for list of available tasks.")

    eval_logger.info(f"Selected Tasks: {task_names}")

    # set datetime before evaluation
    datetime_str = utils.get_datetime_str(timezone=args.timezone)
    if args.output_path:
        if args.log_samples_suffix and len(args.log_samples_suffix) > 15:
            eval_logger.warning("The suffix for log_samples is too long. It is recommended to keep it under 15 characters.")
            args.log_samples_suffix = args.log_samples_suffix[:5] + "..." + args.log_samples_suffix[-5:]

        hash_input = f"{args.model_args}".encode("utf-8")
        hash_output = hashlib.sha256(hash_input).hexdigest()[:6]
        path = Path(args.output_path)
        path = path.expanduser().resolve().joinpath(f"{datetime_str}_{args.log_samples_suffix}_{args.model}_model_args_{hash_output}")
        args.output_path = path

    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"
    
    args.prune_config = parse_prune_args(args)
    args.search_config = parse_search_args(args)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        check_integrity=args.check_integrity,
        show_task_to_terminal=args.show_task_to_terminal,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        cli_args=args,
        predict_only=args.predict_only,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        if args.output_path:
            args.output_path.mkdir(parents=True, exist_ok=True)
            result_file_path = path.joinpath("results.json")
            if result_file_path.exists():
                eval_logger.warning(f"Output file {result_file_path} already exists and will be overwritten.")

            result_file_path.open("w").write(dumped)
            if args.log_samples:
                for task_name, config in results["configs"].items():
                    filename = args.output_path.joinpath(f"{task_name}.json")
                    # Structure the data with 'args' and 'logs' keys
                    data_to_dump = {"args": vars(args), "model_configs": config, "logs": sorted(samples[task_name], key=lambda x: x["doc_id"]), "time": datetime_str}
                    samples_dumped = json.dumps(data_to_dump, indent=4, default=_handle_non_serializable, ensure_ascii=False)
                    filename.open("w", encoding="utf-8").write(samples_dumped)
                    eval_logger.info(f"Saved samples to {filename}")

        return results, samples
    return None, None


def print_results(args, results):
    print(f"{args.model} ({args.model_args}),\ngen_kwargs: ({args.gen_kwargs}),\nlimit: {args.limit},\nnum_fewshot: {args.num_fewshot},\nbatch_size: {args.batch_size}")
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()
