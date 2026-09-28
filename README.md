# bandwidth-reviewer demo

A small messaging-pricing service used to demo the AI code reviewer. Comment
`@bandwidth-reviewer review` on a pull request and a Bedrock model reviews the
diff and posts inline comments.

- `app/`: the sample service (SMS segment pricing, customers)
- `.github/bandwidth-reviewer/review.py`: the reviewer (fetch diff → ask model → post review)
- `.github/workflows/ai-review.yml`: runs the reviewer when someone comments the command

The `add-bulk-send` branch adds a bulk-send feature with a few planted bugs.
Open a PR from it to demo the reviewer.

## Setup (about 15 minutes)

### 1. Check which Bedrock models you can use

The default model is `openai.gpt-oss-120b-1:0` (cheap, strong at code and reasoning).
Check that it's available in your region:

```bash
aws bedrock list-foundation-models --region us-west-2 --query "modelSummaries[?contains(modelId, 'gpt-oss') || contains(modelId, 'qwen3-coder') || contains(modelId, 'nova')].modelId"
```

If it isn't listed, open **Bedrock console → Model access** and enable it, or pick
another model from the list:

| Model | `MODEL_ID` | Why |
| --- | --- | --- |
| gpt-oss 120B (default) | `openai.gpt-oss-120b-1:0` | Cheap, good reasoning, good at code |
| Qwen3 Coder 30B | `qwen.qwen3-coder-30b-a3b-v1:0` | Code-specialized, very cheap |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` | Amazon's own model; widely available |

Model IDs and regions change; trust the command output over this table.

### 2. Try it locally first (dry run, posts nothing)

```bash
pip install -r .github/bandwidth-reviewer/requirements.txt
export GITHUB_TOKEN=$(gh auth token) AWS_REGION=us-west-2
python .github/bandwidth-reviewer/review.py <owner>/<repo> <pr-number> --dry-run
```

### 3. Set up the GitHub Action

1. Get a Bedrock credential, either:
   - **Bedrock API key (simplest):** Bedrock console → **API keys** → **Generate short-term API key**. It only works for Bedrock and expires within 12 hours, so regenerate it before each demo. Or:
   - **IAM access key** for a user or role whose only permission is `bedrock:InvokeModel`.
2. In the repo: **Settings → Secrets and variables → Actions**
   - Secrets: `AWS_BEARER_TOKEN_BEDROCK` (API key), or `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - Variables (optional): `AWS_REGION` (default `us-west-2`), `MODEL_ID`
3. The workflow file must be on the default branch (it is, on `main`).

## Demo script (3 minutes)

1. Open the PR from `add-bulk-send`. Point out it looks reasonable at a glance.
2. Comment `@bandwidth-reviewer review`. 👀 appears on the comment.
3. Show the **Actions** tab while it runs (about 30–60 seconds).
4. The review appears with inline comments. Click **Commit suggestion** on one.
5. Show the footer: which model, how long, how many tokens.

The planted bugs, so you know what a good review should catch:

| File | Bug |
| --- | --- |
| `app/pricing.py` | Segment count changed from rounding up to rounding down, so long messages are undercharged |
| `tests/test_pricing.py` | The two segment tests were edited to match the bug, so the branch's tests still pass |
| `app/bulk_send.py` | Batching loop stops one short, so the last recipient is silently skipped |
| `app/bulk_send.py` | Unknown customer IDs crash with `AttributeError` instead of a clear error |
| `app/bulk_send.py` | Hard-coded API token in source |
| `app/bulk_send.py` | Balance is charged for all recipients even if sending a batch fails partway |
