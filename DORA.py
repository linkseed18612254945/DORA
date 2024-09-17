import sys
import random
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from automatic_prompt_engineer import template, data
from evaluation import GetScore
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from instruction_coupled_kernel import *
import argparse
import openai
import json
import logging
import time

sys.path.append("/tmp/DORA/")

tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

N_INIT = 4
N_ITERATIONS = 5
BATCH_SIZE = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai.api_base = "https://api.chatanywhere.com.cn/v1"
openai.api_key = "sk-4LbX8s4Tlb3UfNhyWEbyZZSFF6qqklBVQXs3sHZpdhMQpjeP"
samples_num = 1 #随机抽取的样例数
instruction_id = [0]#记录指令id
memory = [[],[],[],[],[],[],[],[],[],[]]#保存每个任务实例的反思建议
success_fail = [0,0,0,0,0,0,0,0,0,0]#记录10个任务实例中成功或失败，0表示失败，1表示成功
success_id = [[10000],[10000],[10000],[10000],[10000],[10000],[10000],[10000],[10000],[10000]]#记录每个任务实例成功的具体轮次
class LMForwardAPI:
    def __init__(self, model_name=None, init_prompt=None,
                 prompt_gen_data=None, random_proj=None, intrinsic_dim=None, n_prompt_tokens=None, few_shot_data=None,
                 HF_cache_dir=None, args=None):

        kwargs = {
            'torch_dtype': torch.float16,
            'use_cache': True
        }
        self.ops_model = model_name
        if self.ops_model in ["vicuna"]:
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_cache_dir, low_cpu_mem_usage=True, **kwargs
            ).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                HF_cache_dir,
                model_max_length=1024,
                padding_side="left",
                use_fast=False,
            )
        else:
            raise NotImplementedError

        self.init_prompt_str = init_prompt

        if self.ops_model in ['vicuna']:
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]

        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))

        self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False)
        if self.ops_model == 'vicuna':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['USER:', 'ASSISTANT:']
        else:
            NotImplementedError
        #随机投影（正态分布或者均匀分布）
        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['vicuna']:
                print('Get the embedding firstly to avoid issues')
            else:
                raise NotImplementedError
            mu_hat = np.mean(self.embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(self.embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(intrinsic_dim) * args.sigma)

            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, mu, std)
        elif random_proj == 'uniform':
            for p in self.linear.parameters():
                torch.nn.init.uniform_(p, -1, 1)


        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()


    def eval(self, prompt_embedding=None, init_qa=None, ite=0):
        self.init_token = self.init_prompt_str[0] + init_qa[0]
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = copy.deepcopy(prompt_embedding)
        prompt_embedding = prompt_embedding.type(torch.float32)
        prompt_embedding = self.linear(prompt_embedding)  # 随机投影Az
        prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)


        input_text = f"{self.system_prompt} USER:{self.init_token} ASSISTANT:"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
        input_embed = torch.cat((prompt_embedding, input_embed), 1)
        outputs = self.model.generate(inputs_embeds=input_embed, max_new_tokens=128)
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print('Instruction: {}'.format(instruction))
        with open(args.task + '_instruction.json', 'a') as f:
            json.dump(instruction, f)
        #对指令进行评估得到评分
        dev_perf, instruction_score = GetScore(prompt=instruction, task=args.task, memory=memory, success_fail=success_fail, instruction_id=instruction_id, success_id=success_id, ite=ite)
        dev_perf = dev_perf.sorted()[1][0]
        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_instruction = instruction

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')
        instruction_info = f"instruction: {instruction}, dev_perf: {dev_perf}, instruction_score: {instruction_score}"
        logging.info(instruction_info)
        with open(args.task + '_instruction_info.json', 'a') as f:
            json.dump(instruction_info, f)

        return dev_perf, instruction_score

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set


def prompt_optimizer(args):
    task, HF_cache_dir = args.task, args.HF_cache_dir
    random_proj, intrinsic_dim, n_prompt_tokens = args.random_proj, args.intrinsic_dim, args.n_prompt_tokens
    #prompt_gen_data = ([str([{'thoughts': {'reasoning': 'I need to follow the task instructions and click button ONE, then click button TWO.', 'plan': '- Click button ONE\n- Click button TWO', 'criticism': 'I need to make sure I am accurately clicking the buttons.'}, 'action': {'name': 'click_element', 'input_action_args': {'ref': 1}}}])], [str(['Based on the provided information, the AI assistant\'s strategy was to follow the task instructions and click button ONE, then click button TWO. The assistant planned to click button ONE using the "click_element" action.\n\nHowever, the assistant made a mistake by not including the action to click button TWO in its plan. This was an oversight in the assistant\'s action sequence.\n\nTo devise a new plan of ActionSchema that accounts for this mistake, the assistant should include the action to click button TWO after clicking button ONE. The new plan could be as follows:\n\n[New Plan]\n- Click button ONE\n- Click button TWO\n\nThis new plan ensures that the assistant accurately follows the task instructions and completes the task successfully.'])])
    #subsampled_data = ([str([{'thoughts': {'reasoning': 'I need to follow the task instructions and click button ONE, then click button TWO.', 'plan': '- Click button ONE\n- Click button TWO', 'criticism': 'I need to make sure I am accurately clicking the buttons.'}, 'action': {'name': 'click_element', 'input_action_args': {'ref': 1}}}])],[str(['Based on the provided information, the AI assistant\'s strategy was to follow the task instructions and click button ONE, then click button TWO. The assistant planned to click button ONE using the "click_element" action.\n\nHowever, the assistant made a mistake by not including the action to click button TWO in its plan. This was an oversight in the assistant\'s action sequence.\n\nTo devise a new plan of ActionSchema that accounts for this mistake, the assistant should include the action to click button TWO after clicking button ONE. The new plan could be as follows:\n\n[New Plan]\n- Click button ONE\n- Click button TWO\n\nThis new plan ensures that the assistant accurately follows the task instructions and completes the task successfully.'])])
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    init_prompt = ['\n']
    prompt_gen_template = "[full_DEMO]\n\nThe instruction for reflection was?"
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)


    model_forward_api = LMForwardAPI(model_name=args.model_name, init_prompt=init_prompt,
                                     random_proj=random_proj,
                                     intrinsic_dim=intrinsic_dim, n_prompt_tokens=n_prompt_tokens,
                                     HF_cache_dir=HF_cache_dir, args=args)

    # 随机生成初始N_INIT条soft prompt作为贝叶斯优化初始点
    X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(N_INIT)
    X_return = []
    #对上述贝叶斯优化的初始点进行评估得到reward
    for x in X:
        # 随机抽取反思样例池中的例子
        with open("./reflection_suggestions_pool/"+args.task+"_reflection_example.json", "r", encoding="utf-8") as f:
            content = json.load(f)

        sample_data = random.sample(content, samples_num)
        input = []
        output = []
        for data in sample_data:
            input.append(data["input"])
            output.append(data["output"])
        reference_data = (input, output)
        print(reference_data)
        demos = d_template.fill(reference_data)
        init_qa = [prompt_gen_template.fill(demos)]
        X_return.append(model_forward_api.eval(x, init_qa, 0))


    Y = [X[0] for X in X_return]
    Y_scores = [X[1].squeeze() for X in X_return]

    X = X.to(**tkwargs)
    Y = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
    Y_scores = torch.FloatTensor(np.array(Y_scores)).to(**tkwargs)
    print(f"Best initial point: {Y.max().item():.3f}")

    X_train = X
    y_train = (Y - Y.mean(dim=-2)) / (Y.std(dim=-2))

    # 定义高斯核函数
    matern_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=X_train.shape[-1],
        lengthscale_prior=GammaPrior(3.0, 6.0),
    )
    matern_kernel_instruction = MaternKernel(
        nu=2.5,
        ard_num_dims=Y_scores.shape[-1],
        lengthscale_prior=GammaPrior(3.0, 6.0),
    )

    covar_module = ScaleKernel(
        base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction,
                                         latent_train=X_train.double(), instruction_train=Y_scores))

    # 定义高斯过程回归模型
    gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
    # 定义边缘对数似然
    gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    for i in range(N_ITERATIONS):
        # 根据最大似然原理去拟合代理模型
        fit_gpytorch_model(gp_mll)
        #定义EI采集函数
        EI = ExpectedImprovement(gp_model, best_f=y_train.max().item())

        starting_idxs = torch.argsort(-1 * y_train)[:BATCH_SIZE]
        starting_points = X_train[starting_idxs]

        best_points = []
        best_vals = []
        #探索新的数据点
        for starting_point_for_cma in starting_points:
            if (torch.max(starting_point_for_cma) > 1 or torch.min(starting_point_for_cma) < -1):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI, tkwargs)
            best_points.append(newp)
            best_vals.append(newv)
        print("best_points", best_points)
        print("best_vals", best_vals)
        print(f"best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}")
        #对新探索到的点进行评估得到reward
        for idx in np.argsort(-1 * np.array(best_vals)):
            X_next_point = torch.from_numpy(best_points[idx]).float().unsqueeze(0)

            #随机抽取反思样例池中的例子
            with open("./reflection_suggestions_pool/"+args.task + "_reflection_example.json", "r", encoding="utf-8") as f:
                content = json.load(f)

            sample_data = random.sample(content, samples_num)
            input = []
            output = []
            for data in sample_data:
                input.append(data["input"])
                output.append(data["output"])
            reference_data = (input, output)

            demos = d_template.fill(reference_data)
            init_qa = [prompt_gen_template.fill(demos)]
            X_next_points_return = [model_forward_api.eval(X_next_point, init_qa,i+1)]
            Y_next_point = [X[0] for X in X_next_points_return]

            Y_scores_next_points = [X[1].squeeze() for X in X_next_points_return]

            X_next_point = X_next_point.to(**tkwargs)
            Y_next_point = torch.FloatTensor(Y_next_point).unsqueeze(-1).to(**tkwargs)
            Y_scores_next_points = torch.FloatTensor(np.array(Y_scores_next_points)).to(**tkwargs)

            X = torch.cat([X, X_next_point])
            Y = torch.cat([Y, Y_next_point])
            Y_scores = torch.cat([Y_scores, Y_scores_next_points])

        #根据现有的数据点更新核函数、高斯过程回归模型等
        X_train = X.clone()
        y_train = (Y - Y.mean(dim=-2)) / (Y.std(dim=-2))

        matern_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=X_train.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )

        matern_kernel_instruction = MaternKernel(
            nu=2.5,
            ard_num_dims=Y_scores.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )

        covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel,
                                                                    instruction_kernel=matern_kernel_instruction,
                                                                    latent_train=X_train.double(),
                                                                    instruction_train=Y_scores))

        gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        print(f"Best value found till now: {torch.max(Y)}")

    print('Evaluate on test data...')
    prompts = model_forward_api.return_best_prompt()
    print("Best instruction is:")
    print(prompts)
    return prompts


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='', help='task')
    parser.add_argument('--random_proj', type=str, default='', help='random_proj')
    parser.add_argument('--n_prompt_tokens', type=int, default=5, help='n_prompt_tokens')
    parser.add_argument('--intrinsic_dim', type=int, default=10, help='intrinsic_dim')
    parser.add_argument('--HF_cache_dir', type=str, default='lmsys/vicuna-7b-v1.3', help='HF_cache_dir')
    parser.add_argument('--model_name', type=str, default='vicuna', help='model_name')
    args = parser.parse_args()
    meta_prompt = prompt_optimizer(args)
    success_task = 0
    #统计任务成功率和执行时间
    for i in success_fail:
        if i == 1:
            success_task = success_task + 1
    success_rate = f'Task: {args.task}; Success_rate: {success_task/10}'
    for i in range(10):
        if min(success_id[i])!=10000:
            with open(args.task + '_ENV_trail_Result.txt', 'a') as f:
                f.write(f'Environment #{i} Trial #{min(success_id[i])} : SUCCESS')
                f.write('\n')
        else:
            with open(args.task + '_ENV_trail_Result.txt', 'a') as f:
                f.write(f'Environment #{i} : Fail')
                f.write('\n')

    print(success_rate)
    end_time = time.time()  # 记录结束时间

    duration = end_time - start_time
    print(duration)
    with open('./_IZ_Reflection_Result.txt', 'a') as f:
        f.write(success_rate)
        f.write(str(duration))
        f.write('\n')
