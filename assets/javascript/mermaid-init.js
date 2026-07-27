// Mermaid initialization for MkDocs Material
document.addEventListener('DOMContentLoaded', function() {
  // Check if mermaid is loaded
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true
      }
    });
  }
});

// Re-initialize mermaid when page changes (for instant navigation)
document.addEventListener('DOMContentSwitch', function() {
  if (typeof mermaid !== 'undefined') {
    mermaid.init(undefined, '.mermaid');
  }
});
