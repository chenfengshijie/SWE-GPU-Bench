"""Download pinned runtime tools into node-local scratch space."""
from pathlib import Path
import base64,hashlib,json,shutil,sys,tarfile,urllib.request,zipfile
local=Path(sys.argv[1]).resolve()
bin_dir=local/'bin';bin_dir.mkdir(exist_ok=True)
assets=[
 {'name':'uv','url':'https://files.pythonhosted.org/packages/c3/75/1901383808ba0d421f32060ba0e9884169b685a722e1af5d87fdd4225ba6/uv-0.13.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','digest':'52c7da0fd9056d9551124777ec77de53f0df7017daf8f98a19e947b7837416ed','algorithm':'sha256'},
 {'name':'codex','url':'https://registry.npmjs.org/@openai/codex/-/codex-0.166.0-linux-x64.tgz','digest':'CxDm3+D1cpbGJeHrdzINoG+3wn9BVncP4qpZaaNYkxe8iPtRJOYyCJnnPoVjWRIBvI9vPKbRoifsMSA5xDaUpw==','algorithm':'sha512'}]
assets.append({'name':'opencode','url':'https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.19.5.tgz','digest':'i1qh7VooLGML9xX6ud93y0JA8j/1aPeDYRKjsmYXrJX1TjIWHgBBMkEbEJPZthpWbsJg//XfBgR6Kb7+NPaJiQ==','algorithm':'sha512'})
for asset in assets:
 path=local/(asset['name']+'.download');digest=hashlib.new(asset['algorithm'])
 with urllib.request.urlopen(asset['url'],timeout=180) as response,path.open('wb') as out:
  while data:=response.read(1024*1024):out.write(data);digest.update(data)
 actual=digest.hexdigest() if asset['algorithm']=='sha256' else base64.b64encode(digest.digest()).decode()
 assert actual==asset['digest'],asset['name']+' download checksum mismatch'
 if asset['name']=='uv':
  with zipfile.ZipFile(path) as archive:
   candidates=[n for n in archive.namelist() if n.endswith('/uv')]
   assert len(candidates)==1,candidates
   with archive.open(candidates[0]) as src,(bin_dir/'uv').open('wb') as out:shutil.copyfileobj(src,out)
 elif asset['name']=='opencode':
  with tarfile.open(path,'r:gz') as archive:
   member=next(m for m in archive.getmembers() if m.name.endswith('/bin/opencode') and m.isfile())
   with archive.extractfile(member) as src,(bin_dir/'opencode').open('wb') as out:shutil.copyfileobj(src,out)
 else:
  with tarfile.open(path,'r:gz') as archive:
   for name,suffix in [('codex','vendor/x86_64-unknown-linux-musl/bin/codex'),('codex-code-mode-host','vendor/x86_64-unknown-linux-musl/bin/codex-code-mode-host'),('rg','vendor/x86_64-unknown-linux-musl/codex-path/rg')]:
    member=next(m for m in archive.getmembers() if m.name.endswith(suffix) and m.isfile())
    with archive.extractfile(member) as src,(bin_dir/name).open('wb') as out:shutil.copyfileobj(src,out)
 path.unlink()
for p in bin_dir.iterdir():p.chmod(0o755)
print('NODE_LOCAL_TOOLS_READY',json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bin_dir.iterdir()}),flush=True)
