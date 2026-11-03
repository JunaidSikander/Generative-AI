import React from 'react';
import Layout from '@theme/Layout';
import { FoundationsActivity, PromptActivity, RagActivity, TokensActivity, VectorActivity } from '../components/Activities';

const activities = [
  { id: 'module-1', number: '01', title: 'Classify the AI', description: 'Sort examples into AI, machine learning, deep learning, and generative AI.' },
  { id: 'module-3', number: '03', title: 'Explore tokens and similarity', description: 'Split text into illustrative pieces and compare sample vectors.' },
  { id: 'module-5', number: '05', title: 'Edit a prompt', description: 'Change instructions and compare simulated output previews.' },
  { id: 'module-7', number: '07', title: 'Search a vector collection', description: 'Query six sample records and inspect ranked results.' },
  { id: 'module-8', number: '08', title: 'Guided RAG demo', description: 'Inspect locally retrieved passages and their citations.' },
];

export default function Home() {
  return (
    <Layout title="Optional Interactive Activities" description="Optional browser-based interactive simulations for five modules of the Prompt to Production GenAI course.">
      <main className="activities-page">
        <header className="hero-panel">
          <div className="hero-inner">
            <p className="eyebrow">A companion to the 14-module course</p>
            <h1>Learn by exploring.</h1>
            <p className="hero-copy">These optional, no-account activities add small browser-based experiments alongside the original lessons and Python labs.</p>
            <p className="hero-note"><strong>All results are illustrative simulations.</strong> No AI model, API, backend, or external service is used. For hands-on work with real models, continue with each module’s Python lab.</p>
            <a className="button button--primary hero-button" href="#module-1">Start an activity</a>
          </div>
          <div className="hero-decoration" aria-hidden="true"><span>01</span><span>03</span><span>05</span><span>07</span><span>08</span></div>
        </header>
        <div className="content-wrap">
          <nav className="activity-index" aria-label="Activity index">
            <div className="index-heading"><div><p className="eyebrow">Choose your path</p><h2>Five optional activities</h2></div><span className="index-count">05</span></div>
            <div className="index-grid">{activities.map((activity) => <a className="index-card" href={`#${activity.id}`} key={activity.id}><span className="index-number">{activity.number}</span><strong>{activity.title}</strong><span>{activity.description}</span></a>)}</div>
          </nav>
          <div className="activity-stack">
            <FoundationsActivity />
            <TokensActivity />
            <PromptActivity />
            <VectorActivity />
            <RagActivity />
          </div>
          <aside className="python-path"><div className="python-icon" aria-hidden="true">{ }</div><div><p className="eyebrow">Want to build with real models?</p><h2>The Python labs remain the hands-on path.</h2><p>These activities are simulations for exploration. Use the existing course labs to work with Python packages, local models, and model-provider APIs.</p><a className="lesson-link" href="https://github.com/JunaidSikander/Generative-AI/tree/main/labs">Browse all 14 original labs ↗</a></div></aside>
        </div>
      </main>
    </Layout>
  );
}
