# HDFM Framework - GitHub Repository Setup Guide

This guide shows you how to upload this repository to GitHub and link it to your EcoEvoRxiv preprint.

## Quick Start

### 1. Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and log in
2. Click the "+" icon in top right → "New repository"
3. Settings:
   - **Repository name**: `hdfm-framework`
   - **Description**: "Computational framework for entropy-minimizing forest corridor optimization with backwards climate adaptation"
   - **Visibility**: Public
   - **Initialize**: Do NOT add README, .gitignore, or license (we already have these)
4. Click "Create repository"

### 2. Upload Your Code

Open terminal/command prompt in the `hdfm-framework` directory and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: HDFM framework implementation"

# Connect to GitHub (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/hdfm-framework.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 3. Configure Repository Settings

On your GitHub repository page:

#### Add Topics
Click "⚙️ Settings" → scroll to "Topics"
Add: `forest-management`, `conservation`, `landscape-ecology`, `network-optimization`, `climate-adaptation`, `python`

#### Add Description
Your repository description should read:
"Computational framework for entropy-minimizing corridor network optimization with backwards climate adaptation. Implements algorithms from Hart (2024) EcoEvoRxiv preprint."

#### Enable GitHub Pages (for documentation)
Settings → Pages → Source: Deploy from branch → Branch: main → /docs

### 4. Create Release for Citation

1. Click "Releases" on right sidebar → "Create a new release"
2. Tag version: `v0.1.0`
3. Release title: "HDFM Framework v0.1.0 - Initial Release"
4. Description:
   ```
   Initial release of Hierarchical Dendritic Forest Management (HDFM) Framework
   
   Accompanies preprint:
   Hart, J. (2024). "Hierarchical Dendritic Forest Management: A Vision for 
   Technology-Enabled Landscape Conservation." EcoEvoRxiv. [DOI]
   
   ## Features
   - Entropy-based corridor optimization
   - Dendritic network construction (MST)
   - Backwards climate optimization algorithm
   - Synthetic landscape validation
   - Comprehensive visualization tools
   
   ## Installation
   ```bash
   pip install git+https://github.com/YOUR-USERNAME/hdfm-framework.git
   ```
   
   See README.md for full documentation.
   ```
5. Click "Publish release"

### 5. Get Repository DOI via Zenodo (Optional but Recommended)

1. Go to [Zenodo.org](https://zenodo.org) and log in with GitHub
2. Go to Settings → GitHub → Enable your repository
3. Create a new release on GitHub (triggers Zenodo archival)
4. Zenodo will generate a DOI you can cite

### 6. Update README with Your GitHub URL

Edit `README.md` and replace:
```markdown
git clone https://github.com/yourusername/hdfm-framework.git
```

With:
```markdown
git clone https://github.com/YOUR-ACTUAL-USERNAME/hdfm-framework.git
```

Commit and push the change:
```bash
git add README.md
git commit -m "Update README with correct GitHub URL"
git push
```

## Link to EcoEvoRxiv Preprint

### In Your Preprint Submission

In the "Data Availability" or "Code Availability" section:

```
Code Availability: Complete reference implementation available at 
https://github.com/YOUR-USERNAME/hdfm-framework (v0.1.0). 
Zenodo DOI: [DOI once created]
```

In the preprint text, you can reference:

```
Implementation details and reference code are available in our 
open-source HDFM Framework (https://github.com/YOUR-USERNAME/hdfm-framework).
The framework enables researchers to reproduce our synthetic landscape 
validation and apply the algorithms to their own datasets.
```

### Update Preprint Metadata

On EcoEvoRxiv:
- Add GitHub URL to "Related Identifiers"
- Select relationship type: "IsSupplementedBy"
- Add Zenodo DOI when available

## Repository Checklist

Before making repository public, verify:

- [ ] All code runs without errors
- [ ] Tests pass: `pytest tests/ -v`
- [ ] README is clear and complete
- [ ] LICENSE file is present
- [ ] Examples run successfully
- [ ] GitHub URL is correct in README
- [ ] Contact email is correct
- [ ] CONTRIBUTING.md is complete

## Publicizing Your Repository

### Add Badges to README

Add these badges at the top of README.md:

```markdown
[![DOI](https://zenodo.org/badge/DOI/YOUR-DOI.svg)](https://doi.org/YOUR-DOI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
```

### Share on Social Media

Tweet template:
```
🌲 New #conservation tech: HDFM Framework for climate-adaptive forest corridors

✅ Entropy-minimizing optimization
✅ Backwards climate adaptation
✅ Open-source Python implementation

Code: https://github.com/YOUR-USERNAME/hdfm-framework
Paper: [EcoEvoRxiv link]

#ecology #conservation #ForestManagement
```

### Submit to Relevant Platforms

- EcoLog-L mailing list
- r/ecology, r/conservation on Reddit
- Conservation Evidence
- Methods in Ecology and Evolution blog

## Maintenance

### Responding to Issues

When users open issues:
1. Respond within 1-2 days
2. Label appropriately (bug, feature, question)
3. Thank them for contribution
4. Close resolved issues with explanation

### Accepting Pull Requests

1. Review code for quality
2. Ensure tests pass
3. Check documentation is updated
4. Thank contributor
5. Merge and credit in release notes

### Creating Updates

For significant updates:
```bash
# Make changes
git add .
git commit -m "Add: New feature description"
git push

# Create new release
# Increment version: v0.1.0 → v0.2.0
```

## Support

If you have questions about GitHub setup:
- GitHub Docs: https://docs.github.com
- GitHub Community: https://github.community

For HDFM Framework questions:
- Open an issue: https://github.com/YOUR-USERNAME/hdfm-framework/issues
- Email: viridisnorthllc@gmail.com

---

**Ready to launch?** Follow steps 1-6 above and your framework will be live! 🚀
