# Set up index url for uv
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml << EOF
[[index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
default = true
EOF

# set up uv/uvx auto-completion for zsh
echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> ~/.zshrc

# set up uv link mode in container
echo 'export UV_LINK_MODE="copy"' >> ~/.zshrc

# uv sync to install dependencies
uv sync --all-groups --no-cache
