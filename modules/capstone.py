import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

def show_capstone():
    st.title("Retail Customer Segmentation Analysis")
    
    # Add sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Customer Demographics", "Income & Spending Analysis", "Customer Segmentation"])
    
    # Load data
    df = pd.read_csv("data/capstone.csv")
    
    # Drop CustomerID as it's not analytically useful
    df_analysis = df.drop('CustomerID', axis=1)
    
    if page == "Data Overview":
        data_overview(df, df_analysis)
    elif page == "Customer Demographics":
        customer_demographics(df_analysis)
    elif page == "Income & Spending Analysis":
        income_spending_analysis(df_analysis)
    elif page == "Customer Segmentation":
        customer_segmentation(df, df_analysis)

def data_overview(df, df_analysis):
    st.header("Data Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Dataset Summary")
        st.write(f"• Total Records: {df.shape[0]}")
        st.write(f"• Features: {df.shape[1]}")
        st.write("• Key Variables: Gender, Age, Annual Income, and Spending Score")
    
    st.subheader("Statistical Summary")
    st.dataframe(df_analysis.describe())
    
    st.subheader("Missing Values Check")
    missing = df_analysis.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values found in the dataset!")
    else:
        st.warning("Missing values detected:")
        st.write(missing[missing > 0])

def customer_demographics(df):
    st.header("Customer Demographics Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Gender Distribution")
        gender_counts = df['Gender'].value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index, 
                    title="Gender Distribution", hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Age Distribution")
        fig = px.histogram(df, x="Age", nbins=20, 
                        color_discrete_sequence=['#3366CC'],
                        title="Customer Age Distribution")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig)
    
    # Age statistics by gender
    st.subheader("Age Statistics by Gender")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Age groups
        df['Age Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], 
                                labels=['<25', '25-35', '36-50', '50+'])
        age_gender = df.groupby(['Age Group', 'Gender']).size().reset_index(name='Count')
        fig = px.bar(age_gender, x='Age Group', y='Count', color='Gender', barmode='group',
                    title="Age Group by Gender", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)
    
    with col2:
        # Box plot for age by gender
        fig = px.box(df, y="Age", x="Gender", color="Gender",
                    title="Age Distribution by Gender",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)
    
    with col3:
        st.write("Age Statistics by Gender:")
        st.dataframe(df.groupby('Gender')['Age'].describe().round(1))

def income_spending_analysis(df):
    st.header("Income & Spending Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Annual Income Distribution")
        fig = px.histogram(df, x="Annual Income (k$)", nbins=15,
                        color_discrete_sequence=['#FF6692'],
                        title="Income Distribution")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Spending Score Distribution")
        fig = px.histogram(df, x="Spending Score (1-100)", nbins=15,
                        color_discrete_sequence=['#AB63FA'],
                        title="Spending Score Distribution")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Scatter plot
        fig = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)", 
                        color="Gender", size="Age",
                        title="Income vs. Spending Score",
                        color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)
    
    with col2:
        # Correlation heatmap
        fig, ax = plt.figure(figsize=(6, 4)), plt.axes()
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Correlation Matrix")
        st.pyplot(fig)
    
    # Income and spending by gender
    st.subheader("Income & Spending by Gender")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.box(df, x="Gender", y="Annual Income (k$)", color="Gender",
                    title="Income Distribution by Gender",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)
    
    with col2:
        fig = px.box(df, x="Gender", y="Spending Score (1-100)", color="Gender",
                    title="Spending Score by Gender",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)
    
    # Income and spending by age group
    st.subheader("Income & Spending by Age Group")
    
    if 'Age Group' not in df.columns:
        df['Age Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], 
                                labels=['<25', '25-35', '36-50', '50+'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.box(df, x="Age Group", y="Annual Income (k$)", color="Age Group",
                    title="Income by Age Group",
                    category_orders={"Age Group": ['<25', '25-35', '36-50', '50+']})
        st.plotly_chart(fig)
    
    with col2:
        fig = px.box(df, x="Age Group", y="Spending Score (1-100)", color="Age Group",
                    title="Spending Score by Age Group",
                    category_orders={"Age Group": ['<25', '25-35', '36-50', '50+']})
        st.plotly_chart(fig)

def customer_segmentation(df, df_analysis):
    st.header("Customer Segmentation Analysis")
    
    try:
        # Create a copy of dataframes to avoid modifying originals
        df_cluster = df.copy()
        df_analysis_cluster = df_analysis.copy()
        
        # Encode gender if it exists in the dataset 
        if 'Gender' in df_analysis_cluster.columns:
            # Create a numeric encoding of Gender for analysis purposes
            df_analysis_cluster['Gender_Code'] = df_analysis_cluster['Gender'].map({'Male': 0, 'Female': 1})
        
        # Prepare data for clustering
        features = st.multiselect(
            "Select features for clustering:",
            options=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
            default=['Annual Income (k$)', 'Spending Score (1-100)']
        )
        
        if not features:
            st.warning("Please select at least one feature for clustering.")
            return
        
        num_clusters = st.slider("Number of clusters:", 2, 10, 4)
        
        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_analysis_cluster[features])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to dataframes as integers
        df_analysis_cluster['Cluster'] = cluster_labels.astype(int)
        df_cluster['Cluster'] = cluster_labels.astype(int)
        
        st.subheader("Cluster Analysis")
        
        # Get the color palette for clusters
        colors = px.colors.qualitative.Bold
        cluster_colors = {i: colors[i % len(colors)] for i in range(num_clusters)}
        
        if len(features) == 1:
            # For 1D clustering
            # Use a more distinct color palette to ensure unique colors for each cluster
            fig = px.histogram(df_analysis_cluster, x=features[0], color='Cluster', marginal="box",
                            color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig)
        
        elif len(features) == 2:
            # For 2D clustering
            hover_data = ['Age']
            if 'Gender' in df_analysis_cluster.columns:
                hover_data.append('Gender')
            
            # Use a more distinct color palette to ensure unique colors for each cluster
            color_palette = px.colors.qualitative.Bold + px.colors.qualitative.Vivid + px.colors.qualitative.Set1
            
            fig = px.scatter(df_analysis_cluster, x=features[0], y=features[1], color='Cluster',
                            hover_data=hover_data,
                            title=f"Clusters by {features[0]} and {features[1]}",
                            color_discrete_sequence=color_palette[:num_clusters])
            
            st.plotly_chart(fig)
        
        elif len(features) == 3:
            # For 3D clustering
            hover_data = ['Age']
            if 'Gender' in df_analysis_cluster.columns:
                hover_data.append('Gender')
            
            # Use a more distinct color palette to ensure unique colors for each cluster
            color_palette = px.colors.qualitative.Bold + px.colors.qualitative.Vivid + px.colors.qualitative.Set1
            
            fig = px.scatter_3d(df_analysis_cluster, x=features[0], y=features[1], z=features[2],
                                color='Cluster', hover_data=hover_data,
                                title=f"3D Clusters by {', '.join(features)}",
                                color_discrete_sequence=color_palette[:num_clusters])
            
            st.plotly_chart(fig)
        
        # Cluster Profile Analysis
        st.subheader("Cluster Profiles")
        
        # Select only numeric columns for mean calculation (excluding Cluster itself)
        numeric_cols = df_analysis_cluster.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if 'Cluster' in numeric_cols:
            numeric_cols.remove('Cluster')
        if 'Gender_Code' in numeric_cols:
            numeric_cols.remove('Gender_Code')  # We'll handle gender separately
        
        # Getting mean values for each cluster
        try:
            # Try groupby with agg
            cluster_profile = df_analysis_cluster.groupby('Cluster')[numeric_cols].agg('mean').reset_index()
        except:
            # Fallback method if the first approach fails
            cluster_means = {}
            for col in numeric_cols:
                cluster_means[col] = [df_analysis_cluster[df_analysis_cluster['Cluster'] == i][col].mean() for i in range(num_clusters)]
            
            cluster_profile = pd.DataFrame({
                'Cluster': list(range(num_clusters)),
                **{col: cluster_means[col] for col in numeric_cols}
            })
        
        # Getting counts for each cluster
        cluster_counts = df_cluster['Cluster'].value_counts().sort_index().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        
        # Merge profile and counts
        cluster_profile = pd.merge(cluster_profile, cluster_counts, on='Cluster')
        
        # Calculate percentage
        total = cluster_counts['Count'].sum()
        cluster_profile['Percentage'] = (cluster_profile['Count'] / total * 100).round(1)
        
        # Handle gender breakdown separately if Gender exists
        if 'Gender' in df_cluster.columns:
            try:
                # Method 1: Create a gender breakdown using crosstab
                gender_counts = pd.crosstab(df_cluster['Cluster'], df_cluster['Gender'])
                gender_counts.reset_index(inplace=True)
                
                # Calculate percentages
                for gender in gender_counts.columns[1:]:  # Skip the Cluster column
                    gender_counts[f'{gender}_Pct'] = (gender_counts[gender] / gender_counts.sum(axis=1) * 100).round(1)
                
                # Merge with cluster profile
                cluster_profile = pd.merge(cluster_profile, gender_counts, on='Cluster')
            except:
                # Method 2: Calculate gender distributions manually if first method fails
                gender_breakdown = {}
                gender_cols = df_cluster['Gender'].unique()
                
                for i in range(num_clusters):
                    cluster_data = df_cluster[df_cluster['Cluster'] == i]
                    cluster_size = len(cluster_data)
                    
                    for gender in gender_cols:
                        count = len(cluster_data[cluster_data['Gender'] == gender])
                        pct = (count / cluster_size * 100) if cluster_size > 0 else 0
                        
                        if gender not in gender_breakdown:
                            gender_breakdown[gender] = {'count': [], 'pct': []}
                        
                        gender_breakdown[gender]['count'].append(count)
                        gender_breakdown[gender]['pct'].append(round(pct, 1))
                
                for gender in gender_cols:
                    cluster_profile[gender] = gender_breakdown[gender]['count']
                    cluster_profile[f'{gender}_Pct'] = gender_breakdown[gender]['pct']
        
        # Display the cluster profile information
        for i in range(num_clusters):
            try:
                cluster_data = cluster_profile[cluster_profile['Cluster'] == i]
                percent = cluster_data['Percentage'].values[0] if not cluster_data['Percentage'].empty else 0
                
                with st.expander(f"Cluster {i} Profile ({percent}% of customers)"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if 'Age' in cluster_data.columns:
                            st.write(f"• Avg. Age: {cluster_data['Age'].values[0]:.1f}")
                        if 'Annual Income (k$)' in cluster_data.columns:
                            st.write(f"• Avg. Annual Income: ${cluster_data['Annual Income (k$)'].values[0]:.1f}k")
                        if 'Spending Score (1-100)' in cluster_data.columns:
                            st.write(f"• Avg. Spending Score: {cluster_data['Spending Score (1-100)'].values[0]:.1f}/100")
                    
                    with col2:
                        st.write(f"• Customers in cluster: {cluster_data['Count'].values[0]} ({percent}%)")
                        
                        # Gender breakdown if available
                        if 'Male' in cluster_data.columns and 'Female' in cluster_data.columns:
                            male_pct = cluster_data['Male_Pct'].values[0] if 'Male_Pct' in cluster_data else 0
                            female_pct = cluster_data['Female_Pct'].values[0] if 'Female_Pct' in cluster_data else 0
                            st.write(f"• Gender: {male_pct}% Male, {female_pct}% Female")
            except Exception as e:
                st.error(f"Error displaying cluster {i}: {str(e)}")
        
        # Interpretation
        st.subheader("Cluster Interpretation")
        st.write("""
        Based on the clusters identified, we can interpret customer segments as follows:
        
        1. **Examine the pattern** of each cluster in terms of income and spending behavior
        2. **Look for demographic trends** within clusters (age and gender distributions)
        3. **Compare cluster sizes** to understand market segments
        4. **Consider targeted marketing strategies** for each identified segment
        """)
        
        # Elbow Method for optimal k
        st.subheader("Finding Optimal Number of Clusters")
        
        with st.expander("Elbow Method Analysis"):
            try:
                # Calculate WCSS for different values of k
                wcss = []
                k_range = range(1, 11)
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(scaled_features)
                    wcss.append(kmeans.inertia_)
                
                # Plot the elbow curve
                fig = px.line(x=list(k_range), y=wcss, markers=True,
                            labels={'x': 'Number of Clusters (k)', 'y': 'WCSS (Within-Cluster Sum of Squares)'},
                            title='Elbow Method for Optimal k')
                st.plotly_chart(fig)
                
                st.write("""
                The Elbow Method helps determine the optimal number of clusters:
                
                - Look for the point where the rate of decrease sharply changes
                - This "elbow point" suggests the optimal number of clusters
                - Beyond this point, adding more clusters provides diminishing returns
                """)
            except Exception as e:
                st.error(f"Error calculating the elbow curve: {str(e)}")
    
    except Exception as e:
        st.error(f"An error occurred in customer segmentation: {str(e)}")
        st.write("Please check your data and try again with different parameters.")