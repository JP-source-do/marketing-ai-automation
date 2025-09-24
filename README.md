# 🚀 AI Marketing Automation Platform

An intelligent marketing automation system that provides real-time campaign optimization, performance analytics, and AI-powered insights for digital marketing campaigns across multiple platforms.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## 🌟 Live Demo

🔗 **[View Live Application](https://marketing-ai-automation-wajxhguk3qx3ihgcklsyqq.streamlit.app)**

## ✨ Features

### 📊 **Real-Time Analytics Dashboard**
- Interactive performance metrics visualization
- Multi-platform campaign tracking (Google Ads, Facebook, Instagram, LinkedIn)
- Key performance indicators: ROAS, CTR, conversion rates, CPC
- Dynamic filtering by date range, platform, and campaign

### 🤖 **AI-Powered Insights**
- Intelligent campaign optimization recommendations
- Automated performance analysis and alerts
- Predictive analytics for revenue forecasting
- Budget allocation suggestions based on ML algorithms

### 📈 **Advanced Data Visualization**
- Time-series revenue trending
- Platform performance comparisons
- Interactive charts and graphs using Plotly
- Responsive design for all device types

### 🔧 **Campaign Management**
- Multi-campaign performance monitoring
- Automated reporting and insights generation
- Real-time budget optimization alerts
- Performance benchmarking across platforms

## 🛠️ Technology Stack

- **Frontend**: Streamlit, HTML5, CSS3
- **Backend**: Python 3.13
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Plotly Express
- **AI Integration**: OpenAI API
- **Machine Learning**: Scikit-learn
- **Deployment**: Streamlit Cloud
- **Version Control**: Git, GitHub

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- OpenAI API key (optional, for AI features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JP-source-do/marketing-ai-automation.git
   cd marketing-ai-automation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   # Create .streamlit/secrets.toml for local development
   mkdir .streamlit
   echo 'OPENAI_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📱 Usage

### Dashboard Navigation
1. **Filters Panel**: Use the sidebar to filter data by date range, platforms, and campaigns
2. **Key Metrics**: View real-time performance indicators at the top of the dashboard
3. **Analytics Charts**: Analyze trends with interactive revenue and ROAS visualizations
4. **Campaign Details**: Review detailed performance data in the expandable table
5. **AI Insights**: Get intelligent recommendations and predictions

### AI Features
- Click "Generate AI Campaign Suggestions" for personalized optimization recommendations
- Review automated insights for budget reallocation opportunities
- Monitor performance predictions and growth forecasts

## 📊 Sample Data

The application includes realistic sample data demonstrating:
- **365 days** of campaign performance data
- **4 major platforms**: Google Ads, Facebook, Instagram, LinkedIn
- **Multiple campaigns** with varied performance metrics
- **Realistic ranges** for impressions, clicks, conversions, and revenue

## 🔧 Configuration

### Environment Variables
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your_openai_api_key_here"
GOOGLE_ADS_API_KEY = "your_google_ads_key_here"
FACEBOOK_API_KEY = "your_facebook_key_here"
LINKEDIN_API_KEY = "your_linkedin_key_here"
```

### Customization Options
- Modify `generate_sample_data()` function to connect real data sources
- Customize AI prompts in the insights generation section
- Adjust chart styling and branding in the CSS section
- Add new platforms or metrics as needed

## 🎯 Key Metrics Tracked

| Metric | Description |
|--------|-------------|
| **ROAS** | Return on Ad Spend - Revenue divided by spend |
| **CTR** | Click-Through Rate - Percentage of impressions that result in clicks |
| **CVR** | Conversion Rate - Percentage of clicks that result in conversions |
| **CPC** | Cost Per Click - Average cost for each click |
| **Revenue** | Total revenue generated from campaigns |
| **Spend** | Total advertising spend across all platforms |

## 🤖 AI Integration

The platform integrates with OpenAI's GPT models to provide:
- Intelligent campaign analysis
- Automated optimization suggestions
- Performance prediction algorithms
- Natural language insights generation

## 📈 Future Enhancements

- [ ] Real-time API integrations with ad platforms
- [ ] Advanced machine learning models for prediction
- [ ] Automated campaign optimization execution
- [ ] Email/Slack notifications for performance alerts
- [ ] A/B testing framework integration
- [ ] Custom dashboard creation tools
- [ ] Multi-user access and permissions
- [ ] Advanced reporting and export features

## 🔒 Security

- API keys stored securely using Streamlit secrets management
- No sensitive data stored in repository
- Environment-based configuration for different deployment stages
- HTTPS encryption for all data transmission

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Paul Nicolasora**
- GitHub: [@JP-source-do](https://github.com/JP-source-do)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)
- Email: boyculet1@gmail.com

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- OpenAI for powerful AI capabilities
- Plotly for excellent visualization tools
- The open-source community for inspiration and resources

---

⭐ **If you found this project helpful, please give it a star!** ⭐
