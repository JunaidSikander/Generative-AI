const config = {
  title: 'Prompt to Production — Interactive Activities',
  tagline: 'Optional, browser-based activities for the 14-module GenAI course',
  url: 'https://junaidsikander.github.io',
  baseUrl: '/Generative-AI/',
  organizationName: 'JunaidSikander',
  projectName: 'Generative-AI',
  deploymentBranch: 'gh-pages',
  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  trailingSlash: false,
  presets: [
    [
      'classic',
      {
        docs: false,
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig: {
    navbar: {
      title: 'Prompt to Production',
      items: [
        { to: '/', label: 'Activities', position: 'right' },
        {
          href: 'https://github.com/JunaidSikander/Generative-AI',
          label: 'Course repository',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Course',
          items: [
            {
              label: 'Original Markdown lessons and Python labs',
              href: 'https://github.com/JunaidSikander/Generative-AI',
            },
          ],
        },
      ],
      copyright: 'Interactive learning layer · Simulations run locally in your browser.',
    },
    colorMode: {
      respectPrefersColorScheme: true,
    },
  },
};

module.exports = config;
