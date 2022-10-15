mkdir -p ~/.streamlit/

echo "\
[server]\n\
port= process.env.PORT || 3000\n\
enableCORS = false\n\
headless = true\n\
\n\" > ~/.streamlit/config.toml