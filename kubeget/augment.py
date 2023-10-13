import json
from gpt import gpt
from tqdm import tqdm
import random


example_param_prompts = [
    """
##Input:
#Command: kubectl annotate pods foo description='my frontend'
#Description: Update the annotations on one or more resources.
All Kubernetes objects support the ability to store additional data with the object as annotations. Annotations are key/value pairs that can be larger than labels and include arbitrary string values such as structured JSON. Tools and system extensions may use annotations to store their own data.
Attempting to set an annotation that already exists will fail unless --overwrite is set. If --resource-version is specified and does not match the current resource version on the server the command will fail.
Use "kubectl api-resources" for a complete list of supported resources.
#Syntax: kubectl annotate [--overwrite] (-f FILENAME | TYPE NAME) KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]
#Available Flags:
"flag": "--all-namespaces", "short": "-A", "default": "false", "usage": "If true, check the specified action in all namespaces. "
"flag": "--filename", "short": "-f", "default": "[]", "usage": "Filename, directory, or URL to files identifying the resource to update the annotation "
"flag": "--kustomize", "short": "-k", "default": "", "usage": "Process the kustomization directory. This flag can't be used together with -f or -R. "
"flag": "--output", "short": "-o", "default": "", "usage": "Output format. One of: json|yaml|name|go-template|go-template-file|template|templatefile|jsonpath|jsonpath-as-json|jsonpath-file. "
"flag": "--recursive", "short": "-R", "default": "false", "usage": "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory. "
"flag": "--selector", "short": "-l", "default": "", "usage": "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2). "
#Purpose: Update pod 'foo' with the annotation 'description' and the value 'my frontend' # If the same annotation is set multiple times, only the last value will be applied
##Output:
#Parameters: 
' 
""",
    """
##Input:
#Command: kubectl run nginx --image=nginx
#Description: Create and run a particular image in a pod.
#Syntax: kubectl run NAME --image=image [--env="key=value"] [--port=port] [--dry-run=server|client] [--overrides=inline-json] [--command] -- [COMMAND] [args...]
#Available Flags:
"flag": "--filename", "short": "-f", "default": "[]", "usage": "to use to replace the resource. "
"flag": "--kustomize", "short": "-k", "default": "", "usage": "Process a kustomization directory. This flag can't be used together with -f or -R. "
"flag": "--labels", "short": "-l", "default": "", "usage": "Comma separated labels to apply to the pod(s). Will override previous values. "
"flag": "--output", "short": "-o", "default": "", "usage": "Output format. One of: json|yaml|name|go-template|go-template-file|template|templatefile|jsonpath|jsonpath-as-json|jsonpath-file. "
"flag": "--quiet", "short": "-q", "default": "false", "usage": "If true, suppress prompt messages. "
"flag": "--recursive", "short": "-R", "default": "false", "usage": "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory. "
"flag": "--stdin", "short": "-i", "default": "false", "usage": "Keep stdin open on the container(s) in the pod, even if nothing is attached. "
"flag": "--tty", "short": "-t", "default": "false", "usage": "Allocated a TTY for each container in the pod. "
#Purpose: Start a nginx pod
#Note: Not all Flags are used, only talk about the used ones
##Output:
#Chain of thought: 1) Uses the 'run' subcommand to start a pod\n2) provide the NAME 'nginx' for the pod\n3) Set the base image for the container with the '--image' flag to use the 'nginx' image
""",
]

example_param_prompt = "\n".join(
    [f"###Example {i+1}{p}" for i, p in enumerate(example_param_prompts)]
)

K8S_PARAM_EXTRACTOR = (
    "You are a chatbot, specialized in extracting parameters from commands."
)


def gen_params(dataset):
    with tqdm(
        total=sum(len(x["examples"]) for x in dataset),
        leave=True,
        desc="Generate Params ",
    ) as pbar:
        with open('params.json', "w") as f:
            json.dump(dataset, f)
        return
    
        for data in dataset:
            data = random.choice(dataset)
            command = data["command"]
            description = data["description"]
            syntax = data["syntax"]
            examples = data["examples"]
            flags = data["flags"]

            for example in examples:
                example = random.choice(examples)
                example_description = example["description"]
                example_code = example["code"]
                # prompt = f'Provide the  chain of thought for the following command in the #Command title({example_code}) based on the provided documentation and the examples, reason why the provided command does what the #Description says:\n'
                # prompt += '####Examples:\n'
                # prompt += example_prompt
                # prompt += '####Prompt\n'
                prompt = ""
                prompt += f"##Input:\n"
                prompt += f"#Command: {example_code}\n"
                prompt += f"#Description: {description}\n"
                prompt += f"#Syntax: {syntax}\n"
                prompt += f"#Available Flags:\n{flags}\n"
                prompt += f"#Purpose: {example_description}\n"
                prompt += f"##Output:\n"
                # cot = gpt(K8S_EXPERT, prompt)
                # example['cot'] = cot
                pbar.update(1)

    return dataset


example_cot_prompts = [
    """
##Input:
#Command: kubectl annotate pods foo description='my frontend'
#Description: Update the annotations on one or more resources.
All Kubernetes objects support the ability to store additional data with the object as annotations. Annotations are key/value pairs that can be larger than labels and include arbitrary string values such as structured JSON. Tools and system extensions may use annotations to store their own data.
Attempting to set an annotation that already exists will fail unless --overwrite is set. If --resource-version is specified and does not match the current resource version on the server the command will fail.
Use "kubectl api-resources" for a complete list of supported resources.
#Syntax: kubectl annotate [--overwrite] (-f FILENAME | TYPE NAME) KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]
#Available Flags:
"flag": "--all-namespaces", "short": "-A", "default": "false", "usage": "If true, check the specified action in all namespaces. "
"flag": "--filename", "short": "-f", "default": "[]", "usage": "Filename, directory, or URL to files identifying the resource to update the annotation "
"flag": "--kustomize", "short": "-k", "default": "", "usage": "Process the kustomization directory. This flag can't be used together with -f or -R. "
"flag": "--output", "short": "-o", "default": "", "usage": "Output format. One of: json|yaml|name|go-template|go-template-file|template|templatefile|jsonpath|jsonpath-as-json|jsonpath-file. "
"flag": "--recursive", "short": "-R", "default": "false", "usage": "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory. "
"flag": "--selector", "short": "-l", "default": "", "usage": "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2). "
#Purpose: Update pod 'foo' with the annotation 'description' and the value 'my frontend' # If the same annotation is set multiple times, only the last value will be applied
#Note: Not all Flags are used, only talk about the used ones
##Output:
#Chain of thought: 1) Use the 'annotate' subcommand to add additional data to a kubernetes object\n2) Set the cubernetes object to pods with the 'pods' parameter\n3) Select the pod to add information to by supplying the name of the pod 'foo'\n4)Create a key-value pair with the key as 'description' and the value as 'my frontend' 
""",
    """
##Input:
#Command: kubectl run nginx --image=nginx
#Description: Create and run a particular image in a pod.
#Syntax: kubectl run NAME --image=image [--env="key=value"] [--port=port] [--dry-run=server|client] [--overrides=inline-json] [--command] -- [COMMAND] [args...]
#Available Flags:
"flag": "--filename", "short": "-f", "default": "[]", "usage": "to use to replace the resource. "
"flag": "--kustomize", "short": "-k", "default": "", "usage": "Process a kustomization directory. This flag can't be used together with -f or -R. "
"flag": "--labels", "short": "-l", "default": "", "usage": "Comma separated labels to apply to the pod(s). Will override previous values. "
"flag": "--output", "short": "-o", "default": "", "usage": "Output format. One of: json|yaml|name|go-template|go-template-file|template|templatefile|jsonpath|jsonpath-as-json|jsonpath-file. "
"flag": "--quiet", "short": "-q", "default": "false", "usage": "If true, suppress prompt messages. "
"flag": "--recursive", "short": "-R", "default": "false", "usage": "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory. "
"flag": "--stdin", "short": "-i", "default": "false", "usage": "Keep stdin open on the container(s) in the pod, even if nothing is attached. "
"flag": "--tty", "short": "-t", "default": "false", "usage": "Allocated a TTY for each container in the pod. "
#Purpose: Start a nginx pod
#Note: Not all Flags are used, only talk about the used ones
##Output:
#Chain of thought: 1) Uses the 'run' subcommand to start a pod\n2) provide the NAME 'nginx' for the pod\n3) Set the base image for the container with the '--image' flag to use the 'nginx' image
""",
]

example_cot_prompt = "\n".join(
    [f"###Example {i+1}{p}" for i, p in enumerate(example_cot_prompts)]
)

K8S_EXPERT = (
    "You are a professinal developper that knows kubernetes and the kubectl cli."
)


def gen_cot(dataset):
    with tqdm(
        total=sum(len(x["examples"]) for x in dataset), leave=True, desc="Generate CoT"
    ) as pbar:
        for data in dataset:
            command = data["command"]
            description = data["description"]
            syntax = data["syntax"]
            examples = data["examples"]
            flags = data["flags"]

            for example in examples:
                example_description = example["description"]
                example_code = example["code"]
                prompt = f"Provide the  chain of thought for the following command in the #Command title({example_code}) based on the provided documentation and the examples, reason why the provided command does what the #Description says:\n"
                prompt += "####Examples:\n"
                prompt += example_cot_prompt
                prompt += "####Prompt\n"
                prompt += f"##Input:\n"
                prompt += f"#Command: {example_code}\n"
                prompt += f"#Description: {description}\n"
                prompt += f"#Syntax: {syntax}\n"
                prompt += f"#Available Flags:\n{flags}\n"
                prompt += f"#Purpose: {example_description}\n"
                prompt += f"#Note: Not all Flags are used, only talk about the used ones.\nKeep the answer short and organized with only the cain of tought steps as the output. Use the same notation for the 1) 2) as in the example\n"
                prompt += f"##Output:\n"
                prompt += f"#Chain of thought:"
                cot = gpt(K8S_EXPERT, prompt)
                example["cot"] = cot
                pbar.update(1)

    return dataset


example_qi_prompts = [
    """
##Input:
#Command: kubectl create -f ./pod.json
#Description: Create a resource from a file or from stdin.
JSON and YAML formats are accepted.
#Syntax: kubectl create -f FILENAME
#Available Flags:
"flag": "--filename", "short": "-f", "default": "[]", "usage": "Filename, directory, or URL to files to use to create the resource "
"flag": "--kustomize", "short": "-k", "default": "", "usage": "Process the kustomization directory. This flag can't be used together with -f or -R. "
"flag": "--output", "short": "-o", "default": "", "usage": "Output format. One of: json|yaml|name|go-template|go-template-file|template|templatefile|jsonpath|jsonpath-as-json|jsonpath-file. "
"flag": "--recursive", "short": "-R", "default": "false", "usage": "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory. "
"flag": "--selector", "short": "-l", "default": "", "usage": "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2) "
#Objective: Create a pod using the data in pod.json
##Output:
1. How can I create a pod using the data in the pod.json file?
2. What is the command to create a resource from a file?
3. What flag should I use to specify the file that contains the resource definition?
4. how to create pod from file
5. how to create pod with filename
6. how to start a pod using 'pod.json'
7. create a pod from a file
8. create a pod from pod.json
9. generate new pod from file
10. create pod
""",
    """
##Input:
#Command: kubectl exec mypod -c ruby-container -- date
#Description: Execute a command in a container.
#Syntax: kubectl exec (POD | TYPE/NAME) [-c CONTAINER] [flags] -- COMMAND [args...]
#Available Flags:
"flag": "--container", "short": "-c", "default": "", "usage": "Container name. If omitted, use the kubectl.kubernetes.io/default-container annotation for selecting the container to be attached or the first container in the pod will be chosen "
"flag": "--filename", "short": "-f", "default": "[]", "usage": "to use to exec into the resource "
"flag": "--quiet", "short": "-q", "default": "false", "usage": "Only print output from the remote session "
"flag": "--stdin", "short": "-i", "default": "false", "usage": "Pass stdin to the container "
"flag": "--tty", "short": "-t", "default": "false", "usage": "Stdin is a TTY "
#Objective: Get output from running the 'date' command in ruby-container from pod mypod
##Output:
#Questions:
1. How can I check the current time inside a specific container in a Kubernetes pod?
2. How do I execute a command within a specific container of a pod in Kubernetes?
3. How can I run arbitrary commands within a container running in a Kubernetes pod?
4. how to run command in pod with specific conainer
5. how to execute command in specified pod with container
6. how to start a program inside a container with pods
7. run the date command in the 'ruby-container' in 'mypod'
8. get the date from kubernetes continainer named 'ruby-container'
9. execute command 'date' in the 'ruby-container' in 'mypod'
10. start 'date' in container within a pod
""",
]

example_qi_prompt = "\n".join(
    [f"###Example {i+1}{p}" for i, p in enumerate(example_qi_prompts)]
)
K8S_QI = "You are a professinal developper that knows kubernetes and the kubectl cli."


def gen_qi(dataset):
    "Generate questions and instructions for the data"

    print("k8s question extended:")
    with tqdm(
        total=sum(len(x["examples"]) for x in dataset),
        leave=True,
        desc="Generate questions",
    ) as pbar:
        for data in dataset:
            command = data["command"]
            description = data["description"]
            syntax = data["syntax"]
            examples = data["examples"]
            flags = data["flags"]

            for example in examples:
                example_description = example["description"]
                example_code = example["code"]
                example_cot = example["cot"]
                prompt = f"Generate 3 questions of how to do the objective({example_description}) bellow including some data about the parameters({example_code})\n"
                prompt += "Then generate 3 questions of how to do the same objective in short, informal questions without any capitalization and punctuation\n"
                prompt += "Then generate 4 short instructions to do the objectve including some data about the parameters\n"
                prompt += "In the questions use the values provided in the example codes, but not the flags themselfs\n"
                prompt += "The questions and instructions should be short and numbered from 1 to 10 and humanlike\n"
                prompt += "####Examples:\n"
                prompt += example_qi_prompt
                prompt += "####Prompt\n"
                prompt += f"##Input:\n"
                prompt += f"#Command: {example_code}\n"
                prompt += f"#Description: {description}\n"
                prompt += f"#Syntax: {syntax}\n"
                prompt += f"#Available Flags:\n{flags}\n"
                prompt += f"#Objective: {example_description}\n"
                prompt += f"##Output:\n"
                prompt += f"The 10 questions numbered from 1 to 10, one line each!"
                prompt += f"#Questions:"
                qi = gpt(K8S_QI, prompt, "gpt-3.5-turbo")

                qis = [x.strip() for x in qi.strip().split("\n") if x.strip() != ""]
                qis = [x for x in qis if not x.startswith("#")]
                qis = [x.strip() for x in qis if not x.endswith(":")]
                qis = [
                    x.split(".")[1].strip()
                    if "." in x and x.split(".")[0].isdigit()
                    else x
                    for x in qis
                    if not x.endswith(":")
                ]

                example["questions"] = qis
                pbar.update(1)

    return dataset
