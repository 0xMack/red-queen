<template>
  <!-- The crown and its reflection -- the Looking-Glass queen. Same drawing as public/favicon.svg. -->
  <svg viewBox="0 0 32 32" aria-hidden="true">
    <rect width="32" height="32" rx="7" :fill="palette.queen500" />
    <path d="M7 15.2 L8.2 7.4 L12.6 11.2 L16 5.2 L19.4 11.2 L23.8 7.4 L25 15.2 Z" :fill="palette.fg" />
    <rect x="6" y="16" width="20" height="1.1" :fill="palette.bg" opacity=".55" />
    <path d="M7 17.9 L8.2 25.7 L12.6 21.9 L16 27.9 L19.4 21.9 L23.8 25.7 L25 17.9 Z" :fill="palette.fg" opacity=".32" />
  </svg>
</template>
