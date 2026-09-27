# Cloning the private repo on Colab (fine-grained PAT)

> Status: the repo is currently **public**, so `git clone` needs no token and this
> doc is not needed. Keep it for when you flip the repo back to private before the
> preprint.

Colab needs credentials to clone a private repo. Use a **fine-grained** Personal
Access Token scoped to this one repo, read-only.

## Create the token

1. Open https://github.com/settings/personal-access-tokens/new
   (Settings > Developer settings > Personal access tokens > Fine-grained tokens >
   Generate new token).
2. **Token name:** `colab-kv-transfer-readonly`.
3. **Expiration:** short, e.g. 30 days. Regenerate when it lapses.
4. **Resource owner:** your account (TELLON2001).
5. **Repository access:** "Only select repositories" > pick `kv-transfer-safety`.
6. **Permissions:** Repository permissions > **Contents: Read-only**. That is all a
   clone needs (Metadata read-only is added automatically). Leave everything else at
   "No access". If you later want to `git push` from Colab, set Contents to
   Read and write instead.
7. Generate, and copy the token now (it is shown only once).

## Use it on Colab

**Best: Colab Secrets** (the token never appears in the notebook).

1. In Colab, click the key icon (Secrets) in the left sidebar.
2. Add a secret named `GH_PAT`, paste the token, enable notebook access.
3. In a cell:

   ```python
   from google.colab import userdata
   tok = userdata.get('GH_PAT')
   !git clone https://x-access-token:{tok}@github.com/TELLON2001/kv-transfer-safety.git
   ```

**Quick alternative: prompt at runtime** (not stored in the notebook):

```python
from getpass import getpass
import os
os.environ['GH_PAT'] = getpass('GitHub PAT: ')
!git clone https://x-access-token:$GH_PAT@github.com/TELLON2001/kv-transfer-safety.git
```

Avoid hardcoding the token directly in a cell like
`!git clone https://<TOKEN>@github.com/...`: it lands in the notebook output and
history.

## Hygiene

- Never commit a notebook that contains the token. Clear cell outputs before saving.
- Keep expiry short and Contents read-only.
- Revoke or rotate at https://github.com/settings/tokens?type=beta when done.
- Once the repo is made public (at preprint time), no token is needed to clone.
