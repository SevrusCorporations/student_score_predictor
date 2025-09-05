import py4j.GatewayServer;

public class LinearRegressionEntry {
    private LinearRegression lr;

    public LinearRegressionEntry() {
        lr = new LinearRegression();
    }

    public LinearRegression getLinearRegression() {
        return lr;
    }

    public static void main(String[] args) {
        LinearRegressionEntry app = new LinearRegressionEntry();
        GatewayServer server = new GatewayServer(app);
        server.start();
        System.out.println("Gateway Server Started!");
    }
}
