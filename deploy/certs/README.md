将公网域名 `https://educhat.youwei-ai.com` 挂到当前服务时，请把证书文件放到这个目录。

约定文件名：

- `fullchain.pem`
- `privkey.pem`

默认 `docker-compose.yml` 会把这两个文件挂载到前端 Nginx 容器，并监听：

- `80 -> HTTP`
- `443 -> HTTPS`

如果你的证书文件不在这个目录，可以在部署环境的 `.env` 中覆盖：

- `TLS_CERT_PATH`
- `TLS_KEY_PATH`
