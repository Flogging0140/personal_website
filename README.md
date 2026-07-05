# fizzyvermin.com — link hub

## Edit content
- **Links, socials, bio**: edit `src/data/config.ts`
- **Background photos/videos**: just drop files into `src/assets/backgrounds/` —
  they're picked up automatically (sorted by filename). No code editing needed.
  Supports `.jpg` `.jpeg` `.png` `.webp` `.mp4` `.webm`, mixed freely.
- **Avatar**: replace `public/avatar.jpg` with your headshot (square works best).

## Run locally
```
npm install
npm run dev
```
Opens at http://localhost:4321

## Deploy (one-time setup)
1. Create a new **public** GitHub repo (any name).
2. Push this project to it:
   ```
   git init
   git add .
   git commit -m "init"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
3. In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
   (The workflow in `.github/workflows/deploy.yml` will build and deploy automatically.)
4. In your domain registrar's DNS settings for `fizzyvermin.com`, add:
   - Four `A` records pointing the root domain to GitHub Pages' IPs:
     `185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`
   - (Optional, for `www.fizzyvermin.com`) a `CNAME` record pointing to `YOUR_USERNAME.github.io`
5. Back in **Settings → Pages**, enter `fizzyvermin.com` as the custom domain and enable
   "Enforce HTTPS" once it's verified (can take a few minutes to a few hours).

After that, every `git push` to `main` auto-redeploys.
