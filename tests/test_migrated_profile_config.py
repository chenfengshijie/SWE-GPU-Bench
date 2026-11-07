import tempfile
import unittest
from pathlib import Path

from swe_gpu_bench.shared.config import load_profiles, resolve_profile


class ProfileConfigTests(unittest.TestCase):
    def test_load_profiles_reads_agent_and_text_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "swe_gpu_bench.toml"
            config_path.write_text(
                """
[profiles.agent.codex_default]
framework = "codex"
model = "gpt-5.2"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[profiles.text.oracle_default]
provider = "openai_compatible"
model = "gpt-5.2"
base_url = "https://oracle.example/v1"
api_key_env = "ORACLE_API_KEY"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            profiles = load_profiles(config_path)

        self.assertEqual(profiles["agent.codex_default"].framework, "codex")
        self.assertEqual(profiles["agent.codex_default"].model, "gpt-5.2")
        self.assertEqual(profiles["text.oracle_default"].provider, "openai_compatible")
        self.assertEqual(profiles["text.oracle_default"].base_url, "https://oracle.example/v1")

    def test_resolve_profile_applies_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "swe_gpu_bench.toml"
            config_path.write_text(
                """
[profiles.agent.opencode_kimi]
framework = "opencode"
model = "moonshot/kimi-k2.5"
base_url = "https://moonshot.example/v1"
api_key_env = "MOONSHOT_API_KEY"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            profile = resolve_profile(
                profile_name="opencode_kimi",
                profile_kind="agent",
                config_path=config_path,
                model="deepseek/deepseek-reasoner",
                base_url="https://deepseek.example/v1",
                api_key_env="DEEPSEEK_API_KEY",
            )

        self.assertEqual(profile.framework, "opencode")
        self.assertEqual(profile.model, "deepseek/deepseek-reasoner")
        self.assertEqual(profile.base_url, "https://deepseek.example/v1")
        self.assertEqual(profile.api_key_env, "DEEPSEEK_API_KEY")
