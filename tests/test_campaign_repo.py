from pathlib import Path
import subprocess
from types import SimpleNamespace

from swe_gpu_bench.core import campaign_repo


def test_strict_checkout_fetches_only_base_and_reuses_cache_without_network(tmp_path, monkeypatch):
    source=tmp_path/'source';source.mkdir()
    def git(*args, cwd=source):
        return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
    git('init','-q')
    (source/'value.py').write_text('base\n')
    git('add','.')
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','base')
    base=git('rev-parse','HEAD')
    (source/'value.py').write_text('future answer\n')
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qam','future')
    head=git('rev-parse','HEAD')
    monkeypatch.setenv('GIT_CONFIG_COUNT','1')
    monkeypatch.setenv('GIT_CONFIG_KEY_0','url.'+source.as_uri()+'.insteadOf')
    monkeypatch.setenv('GIT_CONFIG_VALUE_0','https://github.com/fixture/repo.git')
    first=tmp_path/'first';first.mkdir()
    paths=SimpleNamespace(repo_dir=first/'repo')
    campaign_repo.clone_or_update_repo(paths,'fixture/repo',base,cache_root=tmp_path/'cache',strict_commit=True)
    assert git('rev-parse','HEAD',cwd=paths.repo_dir)==base
    assert (paths.repo_dir/'value.py').read_text()=='base\n'
    absent=subprocess.run(['git','cat-file','-e',head],cwd=paths.repo_dir,capture_output=True)
    assert absent.returncode!=0
    original=campaign_repo.run_git
    def no_fetch(args,cwd):
        assert args[1] not in ('fetch','pull')
        return original(args,cwd)
    monkeypatch.setattr(campaign_repo,'run_git',no_fetch)
    second=tmp_path/'second';second.mkdir()
    campaign_repo.clone_or_update_repo(SimpleNamespace(repo_dir=second/'repo'),'fixture/repo',base,
                                      cache_root=tmp_path/'cache',strict_commit=True)
    assert (second/'repo/value.py').read_text()=='base\n'
