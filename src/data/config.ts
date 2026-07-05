// ─────────────────────────────────────────────
// EDIT THIS FILE TO UPDATE YOUR PAGE CONTENT
// ─────────────────────────────────────────────

export const profile = {
  name: "@FizzyVermin",
  tagline: "Adventures & Building stuff.",
  avatar: "/pfp.jpg",
};

// Background media — just drop files into src/assets/backgrounds/.
// They're picked up automatically, sorted by filename. No list-editing needed.
// Mix images (.jpg/.png/.webp) and videos (.mp4/.webm) freely — the rotator handles both.
const imageFiles = import.meta.glob("/src/assets/backgrounds/*.{jpg,jpeg,png,webp}", {
  eager: true,
  query: "?url",
  import: "default",
}) as Record<string, string>;

const videoFiles = import.meta.glob("/src/assets/backgrounds/*.{mp4,webm}", {
  eager: true,
  query: "?url",
  import: "default",
}) as Record<string, string>;

type BackgroundEntry = { type: "image" | "video"; src: string; path: string };

const discovered: BackgroundEntry[] = [
  ...Object.entries(imageFiles).map(([path, src]) => ({ type: "image" as const, src, path })),
  ...Object.entries(videoFiles).map(([path, src]) => ({ type: "video" as const, src, path })),
].sort((a, b) => a.path.localeCompare(b.path));

export const backgroundMedia = discovered.map(({ type, src }) => ({ type, src }));

// How long each background stays before crossfading to the next (ms)
export const backgroundIntervalMs = 6000;

export type LinkItem = {
  label: string;
  href: string;
  sublabel?: string;
  emphasis?: boolean; // true = slightly larger/highlighted "hero" button
};

export const links: LinkItem[] = [
  // {
  //   label: "Join my Discord",
  //   sublabel: "Come hang out",
  //   href: "https://discord.gg/yourinvite",
  //   emphasis: true,
  // },
  {
    label: "Watch my latest YouTube video",
    href: "https://youtube.com/@FizzyVermin",
  },
  {
    label: "Follow on Instagram",
    href: "https://instagram.com/fizzyvermin",
  },
];

export const socials = [
  { label: "YouTube", href: "https://youtube.com/@FizzyVermin", icon: "youtube" },
  { label: "Instagram", href: "https://instagram.com/fizzyvermin", icon: "instagram" },
  // { label: "Discord", href: "https://discord.gg/yourinvite", icon: "discord" },
];
